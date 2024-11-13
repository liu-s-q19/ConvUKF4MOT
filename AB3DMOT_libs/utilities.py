import numpy as np
import scipy.special as sp


def kld_gaussian(mu1, sigma1, mu2, sigma2):
    """Calculate the KL divergence for two multi variable Gaussian distributions"""
    k = len(mu1)
    sigma2_inv = np.linalg.inv(sigma2)
    term1 = np.trace(sigma2_inv @ sigma1)
    term2 = (mu2 - mu1).T @ sigma2_inv @ (mu2 - mu1)
    term3 = np.log(np.linalg.det(sigma2) / np.linalg.det(sigma1))
    kl_div = 0.5 * (term1 + term2 - k + term3)
    return kl_div


def kld_gaussian_1d(mu1, sigma1, mu2, sigma2):
    """Calculate the KL divergence for two Gaussian distributions"""
    kl_div = np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    return kl_div


def kld_beta(alpha1, beta1, alpha2, beta2):
    """Computes the KL divergence between two Beta distributions."""

    # Compute the log Beta values
    log_beta1 = sp.betaln(alpha1, beta1)
    log_beta2 = sp.betaln(alpha2, beta2)

    # Compute the digamma values
    digamma_alpha1 = sp.digamma(alpha1)
    digamma_beta1 = sp.digamma(beta1)
    digamma_sum1 = sp.digamma(alpha1 + beta1)

    # Compute the KL divergence
    kl_div = (log_beta2 - log_beta1 +
              (alpha1 - alpha2) * (digamma_alpha1 - digamma_sum1) +
              (beta1 - beta2) * (digamma_beta1 - digamma_sum1))

    return kl_div


def bisection_method(f, a, b, tol=1e-5):
    """
    Finds a root of the function f within the interval [a, b] using the bisection method.

    :param f: The target function
    :param a: Left endpoint of the interval
    :param b: Right endpoint of the interval
    :param tol: Tolerance (the method stops when the interval width is less than this value)
    :return: An approximate value of the root
    """

    if f(a) * f(b) > 0:
        raise ValueError("The function must have opposite signs at the endpoints of the interval")

    while (b - a) / 2 > tol:
        midpoint = a + (b - a) / 2
        f_mid = f(midpoint)
        if abs(f_mid) < 1e-7:
            return midpoint  # An exact root is found
        elif f(a) * f_mid < 0:
            b = midpoint
        else:
            a = midpoint

    return (a + b) / 2


# def camel_to_snake(name):
#     # 首先将字符串转换为字符列表
#     name_chars = list(name)
#
#     new_chars = []
#
#     for i, char in enumerate(name_chars):
#         if char.isupper() and i != 0 and not name_chars[i - 1].isupper():
#             new_chars.append('_')
#
#         new_chars.append(char.lower())
#
#     return ''.join(new_chars)


def camel_to_snake(name):
    name_chars = list(name)

    new_chars = []

    for i, char in enumerate(name_chars):
        # 检查字符是否大写
        if char.isupper():
            # 如果不是首字符，并且前一个字符不是大写或下一个字符是小写，则在前面添加下划线
            if i != 0 and (not name_chars[i - 1].isupper() or (i + 1 < len(name_chars) and name_chars[i + 1].islower())):
                new_chars.append('_')
            new_chars.append(char.lower())
        else:
            new_chars.append(char)

    return ''.join(new_chars)


def check_jax_availability():
    try:
        # Try importing jax
        import jax.numpy as jnp
        return True
    except ImportError:
        return False