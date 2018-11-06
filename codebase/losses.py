"""
Custom Loss functions in addition to an adversarial loss for distinguishing real samples from fake.

However, minimizing the adversarial loss only won't be ideal as the generator can produce 'real'
samples which are far from the desired temporal value.
Hence need to minimize forecast loss as well.

Additionally, the temporal nature of the data is such that human pose points
travel in a specific direction. Hence, direction prediction is crucial.

Reference: Xingyu Zhou et. al: https://doi.org/10.1155/2018/4907423
           Stephanie L. Hyland et. al: https://arxiv.org/pdf/1706.02633.pdf

"""

class DirectionLoss(object):
    # def __init__(self):
    #     super(DirectionLoss,self).__init__()

    def forward(self, training_input, input, target):
        """
        Calculate Direction loss for temporal prediction.

        Minimixing this is important in pose prediction as the nature of the problem
        implicits that pose points move in a particular direction.

        Calculates direction using the sign function.

        Reference: Xingyu Zhou et. al: https://doi.org/10.1155/2018/4907423

        Args:
            training_input (scalar): Tth point in the input training sequence
            prediction (scalar): predicted (output of the Generator) (T+1)th point
            target (scalar): ground truth (T+1)th point

        Returns:
            scalar: Returns absolute value of the difference in the direction taken by the
                  predicted value as opposed to the target value with respect
                  sto the training sequence.

        """
        sign = lambda x: x and (1, -1)[x < 0]

        return abs(sign(input - training_input) - sign(target - training_input))

class PredictionLoss(object):
    # def __init__(self):
        # super(PredictionLoss,self).__init__()

    def forward(self, input, target, norm='l2'):
        """
        Calculate the Prediction loss for the temporal prediction.
        Determines how far away from the target is the predicted point.
        Can be a L-1 or an L-2 norm.

        Args:
            prediction (scalar): predicted (output of the Generator) (T+1)th point
            target (scalar): ground truth (T+1)th point
            norm (string): determines either an 'l1' or an 'l2' loss

        Returns:
            scalar: Returns an L-1/L-2 prediction loss calculated on (T+1)th predicted value.

        """
        assert norm in ['l1', 'l2']

        if norm == 'l1':
            return abs(target - input)
        else:
            return (target - input)**2

class GeneratorLoss(object):
    def __init__(self, lambda_A=1e-4, lambda_P=1e-4, lambda_D=1e-4, norm='l2'):
        self.lambda_A = lambda_A
        self.lambda_P = lambda_P
        self.lambda_D = lambda_D
        self.norm = norm

        import torch.nn as nn
        self.directionLoss = DirectionLoss()
        self.predLoss = PredictionLoss()
        self.sceLoss = nn.BCEWithLogitsLoss(reduction = 'elementwise_mean')

    def __call__(self, training_input, input, target):
        return self.lambda_A*self.sceLossinput(input, target) + self.lambda_P*self.predLoss(input, target, norm = self.norm) + self.lambda_D*self.directionLoss(training_input, input, target)


if __name__ == '__main__':
    gLoss = GeneratorLoss()
