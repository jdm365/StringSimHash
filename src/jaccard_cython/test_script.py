from jaccard_kernel import cdist_jaccard


if __name__ == '__main__':
    x = [[1, 2, 3], [1, 2, 3, 2345980]]
    y = [[1, 2, 3], [1, 2, 3]]
    print(cdist_jaccard(x, y))
