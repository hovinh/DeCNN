import numpy as np
from keras import backend as K

class Backpropagation():
    def __init__ (self, model, layer_name, input_data, layer_idx = None, masking=None):
		"""
		@params:
			- model: a Keras Model.
			- layer_name: name of layer to be backpropagated, can be determined by 
			model.layers[layer_idx].name. 
			- input_data: a input data to be inspected, must be in proper format 
			to be able to be fed into model.
			- layer_idx: equivalent to layer_name.
			- masking: determine which units in the chosen layer to be backpropagated,
			a numpy array with the same shape with chosen layer.
		"""
        self.model = model
        self.layer_name = layer_name
        self.layer = model.get_layer(layer_name)
        self.input_data = input_data
        
        if (layer_idx is None):
            for i, layer in enumerate(self.model.layers):
                if (layer.name == self.layer_name):
                    self.layer_idx = i
                    break
                    
        if (masking is None):
            shape = [1] + list(self.layer.output_shape[1:])
            masking = np.ones(shape, 'float32')
        self.masking = masking

    def compute(self):
		"""
		@returns:
			- output_data: obtained heatmap.
			- func: a reuseable function to compute backpropagation in the same setting.
		"""
        loss = K.mean(self.layer.output * self.masking)
        gradients = K.gradients(loss, self.model.input)[0]
        func = K.function([self.model.input], [gradients])
        output_data = func([self.input_data])[0] 
        output_data = self.filter_gradient(output_data)
        return (output_data, func)
    
    def filter_gradient(self, x):
		"""
		The gradients to be visualize has non-negative value.
		"""
        x_abs = np.abs(x)
        x_max = np.amax(x_abs, axis=-1)
        return x_max
    

class SmoothGrad(Backpropagation):
    def __init__(self, model, layer_name, input_data, layer_idx = None, masking=None):
		"""
		For parameters, please refer to Backpropagation()
		"""
        super(SmoothGrad, self).__init__(model, layer_name, input_data, layer_idx, masking)

    def compute(self, n_samples=50, batch_size=10):
		"""
		@params:
			- n_samples: number of random sampled to be injected noise and taken average.
			- batch_size: must be <= n_samples. If n_samples is too big, there may be there
			are not enough memories to compute, hence we have to proceed them iteratively 
			batch-by-batch.
		@returns:
			- smooth_gradients: obtained heatmap.
		"""
        _, func = super().compute()
        
        shape = [n_samples] + list(self.model.input.shape[1:])
        new_gradients = np.zeros(shape)
        
        for start_idx in range(0, n_samples, batch_size):
            if (n_samples >= start_idx+batch_size):
                end_idx = start_idx + batch_size
            else:
                end_idx = n_samples
                
            shape = [end_idx-start_idx] + list(self.model.input.shape[1:])
                
            random_noise = np.random.random(shape)
            new_images = random_noise + self.input_data
            gradients = func([new_images])[0]
            new_gradients[start_idx:end_idx, ...] = gradients
        
        smooth_gradients = np.expand_dims(np.mean(new_gradients, axis=0), axis=0)
        smooth_gradients = self.filter_gradient(smooth_gradients)
        return smooth_gradients
    
    
class GuidedBackprop(Backpropagation):
    def __init__(self, model, layer_name, input_data, layer_idx = None, masking=None):
		"""
		For parameters, please refer to Backpropagation()
		"""
        super(GuidedBackprop, self).__init__(model, layer_name, input_data, layer_idx, masking)
     
    def compute(self):
		"""
		@returns:
			- gradients_input: obtained heatmap.
		"""
        forward_values = [self.input_data] + self.feed_forward()
        forward_values_dict = {self.model.layers[i].name:forward_values[i] for i in range(self.layer_idx+1)}
        gradients = self.masking
        
        for layer_idx in range(self.layer_idx-1, -1, -1):
            layer_cur = self.model.layers[layer_idx+1].output
            layer_prev = self.model.layers[layer_idx].output
            layer_prev_name = self.model.layers[layer_idx].name
            
            gradients_cur = gradients
            gate_b = (gradients_cur > 0.) * gradients_cur
            gradients = self.guided_backprop_adjacent(layer_cur,
                                                     layer_prev,
                                                     forward_values_dict[layer_prev_name],
                                                     gate_b)
            if (gradients.min() != gradients.max()):
                gradients = self.normalize_gradient(gradients)
            
        gradients_input = gradients
        gradients_input = self.filter_gradient(gradients_input)
        return gradients_input 
    
    def guided_backprop_adjacent(self, layer_cur, layer_prev, values_prev, gate_b):
        loss = K.mean(layer_cur * gate_b)
        gradients = K.gradients(loss, layer_prev)[0]
        gate_f = K.cast(values_prev > 0., 'float32')
        guided_gradients = gradients * gate_f
        
        func = K.function([self.model.input], [guided_gradients])
        output_data = func([self.input_data])[0]
        return output_data
        
    def feed_forward(self):
        forward_layers = [layer.output for layer in self.model.layers[1:self.layer_idx+1]]
        func = K.function([self.model.input], forward_layers)
        self.forward_values = func([self.input_data])
    
        return self.forward_values 
    
    def normalize_gradient(self, img):
        """
		Gradients computed tend to become pretty small, especially after many layers.
		So after each layer, we will multiply them with a constant to keep them in acceptable 
		range (if applicable).
		"""
        gap = img.max() - img.min()
        if (abs(gap) > 1.):
            return img
        amplitude = 1./gap
        img *= amplitude
        
        return img
    
    
class DeconvNet(GuidedBackprop):
    def __init__(self, model, layer_name, input_data, layer_idx = None, masking=None):
		"""
		For parameters, please refer to Backpropagation()
		"""
        super(DeconvNet, self).__init__(model, layer_name, input_data, layer_idx, masking)

    def compute(self):
		"""
		@returns:
			- gradients_input: obtained heatmap.
		"""
        gradients = self.masking
        
        for layer_idx in range(self.layer_idx-1, -1, -1):
            layer_prev = self.model.layers[layer_idx].output
            layer_cur = self.model.layers[layer_idx+1].output

            forward_values_prev = np.ones([self.input_data.shape[0]] + list(self.model.layers[layer_idx].output_shape[1:])) 
            
            gradients_cur = gradients
            gate_b = (gradients_cur > 0.) * gradients_cur
            gradients = self.guided_backprop_adjacent(layer_cur,
                                                     layer_prev,
                                                     forward_values_prev,
                                                     gate_b)

            if (gradients.min() != gradients.max()):
                gradients = self.normalize_gradient(gradients)
            
        gradients_input = gradients
        gradients_input = self.filter_gradient(gradients_input)
        return gradients_input 