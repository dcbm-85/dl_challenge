"""
Predictor interfaces for the Deep Learning challenge.
"""

from typing import List
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from deep_equation.model import DigitsModel, CalculatorModel
from deep_equation.utils import map_label
class BaseNet:
    """
    Base class that must be used as base interface to implement 
    the predictor using the model trained by the student.
    """

    def load_model(self, model_path):
        """
        Implement a method to load models given a model path.
        """
        pass

    def predict(
        self, 
        images_a: List, 
        images_b: List, 
        operators: List[str], 
        device: str = 'cpu'
    ) -> List[float]:
        """
        Make a batch prediction considering a mathematical operator 
        using digits from image_a and image_b.
        Instances from iamges_a, images_b, and operators are aligned:
            - images_a[0], images_b[0], operators[0] -> regards the 0-th input instance
        Args: 
            * images_a (List[PIL.Image]): List of RGB PIL Image of any size
            * images_b (List[PIL.Image]): List of RGB PIL Image of any size
            * operators (List[str]): List of mathematical operators from ['+', '-', '*', '/']
                - invalid options must return `None`
            * device: 'cpu' or 'cuda'
        Return: 
            * predicted_number (List[float]): the list of numbers representing the result of the equation from the inputs: 
                [{digit from image_a} {operator} {digit from image_b}]
        """
    # do your magic

    pass 


class RandomModel(BaseNet):
    """This is a dummy random classifier, it is not using the inputs
        it is just an example of the expected inputs and outputs
    """

    def load_model(self, model_path):
        """
        Method responsible for loading the model.
        If you need to download the model, 
        you can download and load it inside this method.
        """
        np.random.seed(42)

    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ) -> List[float]:

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            random_prediction = np.random.uniform(-10, 100, size=1)[0]
            predictions.append(random_prediction)
        
        return predictions

class StudentModel(BaseNet):
    """
        load_model: method that loads your best model.
        predict: method that makes batch predictions.
    """
    
    def load_model(self, model_path: str, device = 'cpu'):
        """
        Load the student's trained model.
        """
        model_digits = DigitsModel()
        model_calc = CalculatorModel(model_digits)
        model_calc.load_state_dict(torch.load(model_path, map_location=device))
        return model_calc
    
    def predict(
        self, images_a, images_b,
        operators, device = 'cpu'
    ):
        """Implement this method to perform predictions 
        given a list of images_a, images_b and operators.
        """
        T = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        word_to_ix = dict(zip(['+', '-', '*', '/'],range(4)))
        one_hot = F.one_hot(torch.arange(0, 4), 4)
        label_to_value  = map_label()

        model = self.load_model('resources/calculator_model.pt', device=device)
        model = model.to(device)  
        model.eval()

        predictions = []
        for image_a, image_b, operator in zip(images_a, images_b, operators):            
            
            if operator not in ['+', '-', '*', '/']:
                predictions.append(None)
                continue

            operator =  one_hot[word_to_ix[operator]]
            image_a, image_b, operator = T(image_a).unsqueeze(0), \
                T(image_b).unsqueeze(0), operator.unsqueeze(0)
            image_a, image_b, operator = image_a.to(device), \
                image_b.to(device), operator.to(device)

            output = model(image_a, image_b, operator)
            pred = output.argmax(dim=1, keepdim=True)
            pred = label_to_value[pred.item()]
            if pred == np.Inf:
                pred = np.nan
            predictions.append(pred)
        
        return predictions