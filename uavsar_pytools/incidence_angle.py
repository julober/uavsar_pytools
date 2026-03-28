import numpy as np
import rasterio as rio

def calc_inc_angle(dem, lkv_x, lkv_y, lkv_z, pixel_size=5.556):
    """
    Calculates UAVSAR incidence angle from DEM and look vector components.

    Parameters
    ----------
    dem, lkv_x, lkv_y, lkv_z : np.array or str
        Elevation data and the three components of the look vector.
        Strings are treated as filepaths to be handled by rasterio.
    pixel_size : float
        Pixel size of all components in [m]. Default value is for 
        UAVSAR images from JPL.
    
    Returns
    -------
    inc : np.array
        Incidence angle in degrees.
    """
    # Calculate gradient of DEM
    if type(dem) == str:
        with rio.open(dem) as src:
            dem_arr = src.read(1).astype(np.float32)
            row_grad, col_grad = np.gradient(dem_arr, pixel_size)
            dem_shape = dem_arr.shape
    elif type(dem) == np.ndarray:
        row_grad, col_grad = np.gradient(dem.astype(np.float32), pixel_size)
        dem_shape = dem.shape
    else:
        raise ValueError('Pass filepath or np.array for DEM data.')

    # Map numpy row/col gradients to geographic x/y gradients
    dx = -col_grad
    dy = row_grad  

    # Calculate true UNIT surface normal vectors: (-dx, -dy, 1)
    norm_mag = np.sqrt(dx**2 + dy**2 + 1.0)
    n_x = -dx / norm_mag
    n_y = -dy / norm_mag
    n_z = 1.0 / norm_mag

    # Look vectors
    lkv = {}
    components = [lkv_x, lkv_y, lkv_z]
    directions = ['x','y','z']

    for comp_idx, vector in enumerate(components):
        if type(vector) == str:
            with rio.open(vector) as src:
                lkv[directions[comp_idx]] = src.read(1)
        elif type(vector) == np.ndarray:
            assert vector.shape == dem_shape, 'Look vector data must be the same shape as DEM data.'
            lkv[directions[comp_idx]] = vector
        else:
            raise ValueError('Pass filepath or np.array for DEM data.')
        
    # Calculate look vector magnitude
    # lkv_mag = np.zeros_like(lkv['x'])
    # for direction, arr in lkv.items():
    #     lkv_mag = lkv_mag + arr**2
    # lkv_mag = lkv_mag**0.5
    # lkv_mag[lkv_mag == 0] = np.nan
    # # Unit vectors
    # unit_lkv = {}
    # for direction, arr in lkv.items():
    #     unit_lkv[direction] = -arr/lkv_mag
    lkv_mag = np.sqrt(lkv['x']**2 + lkv['y']**2 + lkv['z']**2)
    lkv_mag[lkv_mag == 0] = np.nan

    unit_lkv = {}
    for direction in directions:
        unit_lkv[direction] = -(lkv[direction] / lkv_mag)

    # Calculate Incidence Angle (Dot Product)
    # Since both vectors are normalized, dot product = cos(theta)
    inc_cos = (unit_lkv['x'] * n_x) + (unit_lkv['y'] * n_y) + (unit_lkv['z'] * n_z)
    
    # Clip to exactly [-1.0, 1.0] to prevent floating point errors from crashing arccos
    inc_cos = np.clip(inc_cos, -1.0, 1.0)
    inc = np.arccos(inc_cos)

    return np.rad2deg(inc)
    