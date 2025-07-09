"""
Custom validators for BCI Communicator
"""
from django.core.exceptions import ValidationError
from bci.models import TrainedModel


def validate_motor_imagery_model(model_id):
    """Validate that model is suitable for motor imagery"""
    try:
        model = TrainedModel.objects.get(id=model_id)
        if model.approach != 'motor_imagery':
            raise ValidationError('Model must be a motor imagery model')
        if not model.is_active:
            raise ValidationError('Model must be active')
        if model.n_classes != 4:
            raise ValidationError('Motor imagery model must have 4 classes')
        return model
    except TrainedModel.DoesNotExist:
        raise ValidationError('Motor imagery model does not exist')


def validate_p300_model(model_id):
    """Validate that model is suitable for P300"""
    try:
        model = TrainedModel.objects.get(id=model_id)
        if model.approach != 'p300':
            raise ValidationError('Model must be a P300 model')
        if not model.is_active:
            raise ValidationError('Model must be active')
        return model
    except TrainedModel.DoesNotExist:
        raise ValidationError('P300 model does not exist')


def validate_vocabulary_completeness(vocabulary, available_letters):
    """Validate that all vocabulary words can be spelled"""
    available_set = set(letter.upper() for letter in available_letters)
    
    for word in vocabulary:
        word_letters = set(word.upper())
        if not word_letters.issubset(available_set):
            missing = word_letters - available_set
            raise ValidationError(
                f'Word "{word}" contains unavailable letters: {missing}'
            )