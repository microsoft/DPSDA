from .api import API


def get_api_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == 'DALLE':
        from .dalle_api import DALLEAPI
        return DALLEAPI
    elif name == 'stable_diffusion':
        from .stable_diffusion_api import StableDiffusionAPI
        return StableDiffusionAPI
    elif name == 'improved_diffusion':
        from .improved_diffusion_api import ImprovedDiffusionAPI
        return ImprovedDiffusionAPI
    else:
        raise ValueError(f'Unknown API name {name}')


__all__ = ['get_api_class_from_name', 'API']
