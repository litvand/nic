def plain_sgd():
    coefs = WeightVector()
    for input, target in dataset:
        output = coefs.dot(input) + intercept
        # clip derivative with large values to avoid numerical instabilities
        deriv_of_loss = clamp(derivative_of_hinge_loss(output, target), min=-MAX_DERIV, max=MAX_DERIV)

        coefs *= clamp(1.0 - ((1-l1_ratio) * lr * alpha), min=0)
        coefs -= lr * (x * deriv_of_loss)

        if fit_intercept:
            intercept_update = deriv_of_loss + (2*alpha if one_class else 0)
            intercept -= lr * intercept_update * intercept_decay

# For one-class:
l1_ratio = 0
nu = 0.5
alpha = nu/2

# For non-sparse datasets:
intercept_decay = 1.0
