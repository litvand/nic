def plain_sgd():
    coefs = WeightVector()
    for input, target in dataset:
        output = coefs.dot(input) + intercept
        deriv_of_loss = derivative_of_hinge_loss(output, target)

        coefs *= clamp(1.0 - lr * alpha, min=0)
        coefs -= lr * (x * deriv_of_loss)

        intercept_update = deriv_of_loss + (2 * alpha if one_class else 0)
        intercept -= lr * intercept_update


nu = 0.5
alpha = nu / 2


self.offset_ = 1 - np.atleast_1d(intercept)
svm_output = coefs.dot(input) - self.offset_ > 0
#  = coefs.dot(input) + intercept - 1 > 0
#  = coefs.dot(input) + intercept > 1
