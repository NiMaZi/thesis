import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

def eval_embeddings(get_emb, triplets, dist=cosine, inspect=False):
    
    hits = 0
    known = 0
    for triplet in triplets:
        emb1 = get_emb(triplet[0])
        emb2 = get_emb(triplet[1])
        emb3 = get_emb(triplet[2])
        
        if all(emb is not None for emb in (emb1, emb2, emb3)):
            known += 1
            if dist(emb1, emb2) < dist(emb1, emb3):
                hits += 1
            if dist(emb1, emb2) < dist(emb2, emb3):
                hits += 1

    accuracy = hits / (len(triplets) * 2)
    known_rate = known / len(triplets)
    known_accuracy = hits / (known * 2)
            
    if inspect:
        return accuracy, known_rate, known_accuracy
    
    return accuracy


def eval_plot(results, path, samples='Tokens', known=True):
    r = list(results.values())
    r.sort(key=lambda d: d['samples'])
    tokens = [d['samples'] for d in r]
    acc = [d['accuracy'] for d in r]
    size = (4, 4)
    main_plot = 111
    
    if known:
        known_acc = [d['known_accuracy'] for d in r]
        size = (6, 3)
        main_plot = 121

    fig = plt.figure(figsize=size)

    ax = fig.add_subplot(main_plot)
    ax.set_title('Accuracy')
    ax.set_xlabel(samples)
    ax.set_ylabel('Accuracy')
    ax.semilogx(tokens, acc)

    if known:
        ax = fig.add_subplot(122)
        ax.set_title('Accuracy for known words')
        ax.set_xlabel(samples)
        ax.set_ylabel('Known accuracy')
        ax.semilogx(tokens, known_acc)

    plt.tight_layout()
    plt.show()
    
    fig.savefig(path)