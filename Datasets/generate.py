import numpy as np

def generate_points(radius, target_bytes):
    # Each row has 2 float64 => 16 bytes per row
    bytes_per_row = 16
    approx_rows = target_bytes // bytes_per_row

    # Uniformly distributed points inside a circle using polar coordinates
    r = radius * np.sqrt(np.random.uniform(0, 1, approx_rows))
    theta = np.random.uniform(0, 2 * np.pi, approx_rows)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    points = np.column_stack((x, y))
    
    print(f"Generated {points.shape[0]:,} rows (~{points.nbytes / (1024**2):.2f} MB)")
    return points

if __name__ == "__main__":
    radius = 2
    target_size_mb = 300
    target_bytes = target_size_mb * 1024 * 1024

    points = generate_points(radius, target_bytes)

    # Save to CSV with header
    np.savetxt("circular_points.csv", points, delimiter=",", fmt="%.6f", header="x,y", comments='')
    print("Saved to 'circular_points.csv'")
