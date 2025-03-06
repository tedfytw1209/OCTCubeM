# Copyright (c) Zixuan Liu et al, OCTCubeM group
# All rights reserved.


# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import ttach as tta

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from typing import Dict

class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        print('grads', grads.shape, 'activations', activations.shape, 'input_tensor', input_tensor.shape)
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        print('weights', weights.shape, 'activations', activations.shape)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")
        print(weighted_activations.shape)
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:

        if isinstance(input_tensor, Dict):
            pass
        else:
            input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        print(len(self.activations_and_grads.gradients), len(self.activations_and_grads.activations))
        print( self.activations_and_grads.activations[0].shape)
        print(self.uses_gradients)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            print(outputs.cpu().data.numpy())
            print('target_categories', target_categories)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        if isinstance(input_tensor, Dict):
            cam_text1 = cam_per_layer[-2]
            cam_text2 = cam_per_layer[-1]
            cam_per_layer = cam_per_layer[:-2]
            return [self.aggregate_multi_layers(cam_per_layer), cam_text1, cam_text2, outputs.detach().cpu().numpy()]
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            # depth, width, height = input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        if isinstance(input_tensor, Dict):
            target_size_image = self.get_target_width_height(input_tensor['image'])
            target_size_text1 = self.get_target_width_height(input_tensor['text1'])
            target_size_text2 = self.get_target_width_height(input_tensor['text2'])
            print('input_tensor image', input_tensor['image'].shape, 'target_size', target_size_image)
            print('input_tensor text1', input_tensor['text1'].shape, 'target_size', target_size_text1)
            print('input_tensor text2', input_tensor['text2'].shape, 'target_size', target_size_text2)
            cam_per_target_layer = []
            # Loop over the saliency image from every layer
            for i in range(len(self.target_layers)):
                target_layer = self.target_layers[i]
                layer_activations = None
                layer_grads = None
                if i < len(activations_list):
                    layer_activations = activations_list[i]
                if i < len(grads_list):
                    layer_grads = grads_list[i]
                # print('layer_activations:', layer_activations.shape, 'layer_grads:', layer_grads.shape)
                if i == 0:
                    cam_image = self.get_cam_image(input_tensor['image'], target_layer, targets, layer_activations, layer_grads, eigen_smooth)
                    cam_image = np.maximum(cam_image, 0)
                    print('cam_image shape:', cam_image.shape, 'target_size shape:', target_size_image)
                    if len(target_size_image) == 3:
                        target_size = target_size_image[:-1]
                        # target_size = target_size_image[1:]
                        # cam = torch.repeat_interleave(torch.tensor(cam), 3, dim=0)
                        # interpolate the first dimensionto the original size
                        cam_reshaped = torch.tensor(cam_image).permute(1, 2, 0)  # Convert to (16, 16, 20)

                        # Use interpolate for upsampling, the goal is to change the last dimension from 20 to 60
                        cam_reshaped = cam_reshaped.float()
                        upsampled_cam = F.interpolate(cam_reshaped, size=(60), mode='linear').squeeze(0)

                        # Reshape the tensor back to the original dimension order
                        cam = upsampled_cam.permute(2, 0, 1)  # Convert to (60, 16, 16)

                        cam = cam.numpy()


                    if len(cam.shape) == 5:

                        pass
                    print('cam shape:', cam.shape, target_size, input_tensor['image'][0].shape, input_tensor['text1'][0].shape, input_tensor['text2'][0].shape)



                    scaled = scale_cam_image(cam, target_size)
                    cam_per_target_layer.append(scaled[:, None, :])
                elif i >= 1:
                    textname = 'text' + str(i)
                    print('target_layer:', target_layer, 'textname:', textname)
                    cam_text1 = self.get_cam_image(input_tensor[textname], target_layer, targets, layer_activations, layer_grads, eigen_smooth)
                    cam = np.maximum(cam_text1, 0)
                    target_size = target_size_text1
                    scaled = scale_cam_image(cam, target_size)
                    cam_per_target_layer.append(scaled[:, None, :])

            return cam_per_target_layer

        target_size = self.get_target_width_height(input_tensor)
        print('input_tensor', input_tensor.shape, 'target_size', target_size)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            print(cam.shape, target_size)
            if len(target_size) == 3:
                target_size = target_size[:-1]

                # interpolate the first dimensionto the original size
                cam_reshaped = torch.tensor(cam).permute(1, 2, 0)  # Convert to (16, 16, 20)

                # Use interpolate for upsampling, the goal is to change the last dimension from 20 to 60
                cam_reshaped = cam_reshaped.float()
                upsampled_cam = F.interpolate(cam_reshaped, size=(60), mode='linear').squeeze(0)

                # Reshape the tensor back to the original dimension order
                cam = upsampled_cam.permute(2, 0, 1)  # Convert to (60, 16, 16)

                cam = cam.numpy()
                input_tensor = input_tensor[0]
                print(cam.shape, target_size, input_tensor.shape)
            if len(cam.shape) == 5:

                pass
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
