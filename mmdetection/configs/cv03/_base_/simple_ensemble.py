

def ensemble(models, model_input): 
    outputs = [model.outputs[0] for model in models] 
    y = Average()(outputs) 
    model = Model(model_input, y, name='ensemble') 
    return model 
    
ensemble_model = ensemble(models, model_input)