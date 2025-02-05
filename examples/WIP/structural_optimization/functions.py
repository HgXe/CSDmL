import numpy as np

def map_parametric_wing_pressure(coords):
    # convert the input coordinates to the output coordinates
    # input coords are (index, u, v)
    # output are (u', v')
    
    # Indices
    # 11 ─ Right wing bottom
    # 12 ─ Right wing top
    # 19 ─ Left wing bottom
    # 20 ─ Left wing top

    #   map of input coordinates
    #   o is the origin of each index
    #   x is the output origin
    #   u is horizontal, v is vertical (sometimes flipped)
    #   ┌──────────┐o──────────┐    (TE)
    #   │          ││          │
    #   │    19    ││    11    │    bottom
    #   │          ││          │
    #   └──────────o└──────────┘    (LE)
    #   ┌──────────┐o──────────┐    (LE)
    #   │          ││          │
    #   │    20    ││    12    │    top
    #   │          ││          │
    #   x──────────o└──────────┘    (TE)

    out = np.zeros((len(coords), 2))

    for i, coord in enumerate(coords):
        ind, uv = coord
        if len(uv.shape) == 1:
            uv = uv.reshape(1, -1)
        u, v = uv[:, 0], uv[:, 1]
        if ind == 11:
            out[i,:] = np.hstack((u*0.5+0.5, 1-v*0.5))
        elif ind == 12:
            out[i,:] = np.hstack((u*0.5+0.5, 0.5-v*0.5))
        elif ind == 19:
            out[i,:] = np.hstack((0.5-u*0.5, 0.5+v*0.5))
        elif ind == 20:
            out[i,:] = np.hstack((0.5-u*0.5, v*0.5))

    return out


if __name__ == "__main__":
    # Test the function
    coords = [(11, (0, 0)), (12, (0, 0)), (19, (0, 0)), (20, (0, 0))]
    print(map_parametric_wing_pressure(coords))
    # Expected output: [(0.5, 1), (0.5, 0.5), (0.5, 0.5), (0.5, 0)]

    coords = [(11, (1, 1)), (12, (1, 1)), (19, (1, 1)), (20, (1, 1))]
    print(map_parametric_wing_pressure(coords))
    # Expected output: [(1, 0.5), (1, 0), (0, 1), (0, 0.5)]