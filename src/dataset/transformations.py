# Augment the data by applying transformations to train and test set
from torchvision import transforms


im_size = 128
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def transforms_inv(im_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.Resize((im_size, im_size)),
            transforms.RandomRotation(20),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def train_transforms(im_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.RandomRotation(20),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def test_transforms(im_size):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
