import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def tf_fminbound(func, x1, x2, args=(), xtol=1e-5, maxfun=500):
    """
    bounded minimization for scalar functions (TensorFlow implemention based on
    fminbound from scipy.optimize:
    https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/optimize/optimize.py#L1675.
    :param func: function which returns value to be minimized
    :param x1: lower optimization limit
    :param x2: higher optimization limit
    :param args: extra arguments passed to function
    :param xtol: convergence tolerance
    :param maxfun: maximum number of function evaluations
    :return: parameter value which minimizes objective function
    """
    x1 = tf.keras.backend.cast_to_floatx(x1)
    x2 = tf.keras.backend.cast_to_floatx(x2)
    xatol = tf.keras.backend.cast_to_floatx(xtol)
    sqrt_eps = tf.keras.backend.cast_to_floatx(tf.sqrt(2.2e-16))
    golden_mean = tf.keras.backend.cast_to_floatx(0.5 * (3.0 - tf.sqrt(5.0)))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = tf.keras.backend.cast_to_floatx(0.0)
    x = xf
    fx = func(x, *args)
    num = 1

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * tf.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1
    while tf.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = True
        # Check for parabolic fit
        if tf.abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = tf.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if (tf.abs(p) < tf.abs(0.5 * q * r)) and (p > q * (a - xf)) and (
                    p < q * (b - xf)):

                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = tf.sign(xm - xf) + tf.keras.backend.cast_to_floatx(
                        (xm - xf) == 0)
                    rat = tol1 * si
            else:  # do a golden section step
                golden = True
        if golden:  # Do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e

        si = tf.sign(rat) + tf.keras.backend.cast_to_floatx(rat == 0)
        x = xf + si * tf.maximum(tf.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * tf.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxfun:
            break
    return xf
