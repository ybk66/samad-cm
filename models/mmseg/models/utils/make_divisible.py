def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """
    """这是一个名为make_divisible.py的程序文件，它包含一个名为make_divisible的函数。该函数用于将通道数舍入为最接近能够被除数整除的值。此函数来自于TensorFlow代码库，并用于确保所有层的通道数都可以被指定的除数整除。函数具有四个参数：value（原始通道数）、divisor（除数）、min_value（输出通道的最小值，默认为除数）、min_ratio（舍入通道数与原始通道数的最小比例，默认为0.9）。函数返回修改后的通道数值。"""

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
