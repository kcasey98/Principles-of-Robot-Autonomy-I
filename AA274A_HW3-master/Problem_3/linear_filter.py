#!/usr/bin/env python3

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########

    k = F.shape[0]
    ell = F.shape[1]
    c = F.shape[2]

    m = I.shape[0]
    n = I.shape[1]

    G = np.zeros((m,n))

    I_pad = np.pad(I, ((k//2, k//2), (ell//2, ell//2), (0,0)), constant_values=0)
    # I_pad = np.pad(I, ((0, k), (0, ell), (0, 0)), mode='constant')
    # I_pad = np.pad(I, ((1, k-1), (1, ell-1), (0, 0)), mode='constant')
    # I_pad = np.pad(I, pad_width=1)

    for i in range(m):
        for j in range(n):
            G[i,j] = np.dot(F.flatten(),I_pad[i:i+k,j:j+ell].flatten())

    print("G", G)
    print("G shape", G.shape)

    print("GG", G[0:G.shape[0]-1,0:G.shape[1]-1])
    # print("G shape", G.shape)

    # G = G[0:G.shape[0]-1,0:G.shape[1]-1]

    return G

    # raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########

    print("normmm")
    k = F.shape[0]
    ell = F.shape[1]
    c = F.shape[2]
    m = I.shape[0]
    n = I.shape[1]

    G = np.zeros((m,n))

    I_pad = np.pad(I, ((k//2, k//2), (ell//2, ell//2), (0,0)), constant_values=0)
    # I_pad = np.pad(I, [(0, k), (0, ell), (0, 0)], mode='constant')

    for i in range(m):
        for j in range(n):
            top = np.dot(F.flatten(),I_pad[i:i+k,j:j+ell].flatten())
            bottom = np.linalg.norm(F.flatten())*np.linalg.norm(I_pad[i:i+k,j:j+ell].flatten())
            G[i,j] = top/bottom

    print("G", G)
    print("G shape", G.shape)

    print("GG", G[0:G.shape[0]-1,0:G.shape[1]-1])

    return G

    raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 3, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()

        corr_img = corr(filt, test_card)
        # corr_img = norm_cross_corr(filt, test_card)

        # print("norm",norm_cross_corr(filt, test_card))

        stop = time.time()
        print('Correlation function runtime:', stop - start, 's')
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
