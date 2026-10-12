import matplotlib.pyplot as plt

def plot(xs: list[float], ys: list[float], name_x: str,
         name_y: str, r: float) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, color='hotpink', edgecolor='black')
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(f'{name_x} vs {name_y} (r = {r:.2f})')
    plt.show()
