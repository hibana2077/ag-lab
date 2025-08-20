import matplotlib.pyplot as plt


def plot_loss(history, out_path):
    if not history:
        return
    epochs, vals = zip(*history)
    plt.figure()
    plt.plot(epochs, vals, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ppl(history, out_path):
    if not history:
        return
    epochs, vals = zip(*history)
    plt.figure()
    plt.plot(epochs, vals, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distance_curve(distance2acc, out_path):
    if not distance2acc:
        return
    items = sorted(distance2acc.items())
    dist, acc = zip(*items)
    plt.figure()
    plt.plot(dist, acc, marker='o')
    plt.xlabel('Gap Distance')
    plt.ylabel('Hit Rate')
    plt.title('Needle Hit Rate by Gap')
    plt.savefig(out_path, dpi=150)
    plt.close()
