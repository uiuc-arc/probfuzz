import numpy as np
import json

varCount = 0
_floatinfo = np.finfo(np.float64)
_intinfo = np.iinfo(np.int64)
_float_special_values = [0.0, 1.0, _floatinfo.min, _floatinfo.max, _floatinfo.max - 1.0, _floatinfo.min + 1.0,
                         _floatinfo.eps,
                         _floatinfo.tiny, 0.00001, -0.00001]
int_special_values = [0, 1, _intinfo.min, _intinfo.max, _intinfo.min + 1, _intinfo.max - 1]

_FUZZERTYPE = 'UNSTR'


def is_unstructured():
    return _FUZZERTYPE == 'UNSTR'


def set_fuzzer_type(type):
    global _FUZZERTYPE
    _FUZZERTYPE = type


def getSupportedDistributions(support, models=None, pps="name"):
    if support is None:
        return None

    if models is None:
        models = parse_models()

    return [model for model in models if pps in model and includes(model["support"], support)]


def get_special_values(datatype):
    if datatype == 'i':
        return np.random.choice(int_special_values)
    elif datatype == 'f':
        return np.random.choice(_float_special_values)
    elif datatype == 'i+':
        arr = [x for x in int_special_values if x != 0]
        return np.abs(np.random.choice(arr))
    elif datatype == "f+":
        arr = [x for x in _float_special_values if x > 0.0]
        return np.abs(np.random.choice(arr))
    elif datatype == "0f+":
        arr = [x for x in _float_special_values if x >= 0.0]
        return np.abs(np.random.choice(arr))
    elif datatype == 'p':
        return np.random.choice([0.0, 1.0, 0.5])
    elif datatype == '(0,1)':
        return np.random.choice([_floatinfo.eps, _floatinfo.tiny])
    elif datatype == '0i+':
        return np.abs(np.random.choice(int_special_values))
    else:
        print('Unexpected type ' + datatype)
        exit(-1)


def generate_primitives(data_type, size=1, is_special=False):
    if is_special and data_type != 'b':
        x_data = np.array([get_special_values(data_type) for _ in range(0, size)])
    else:
        if data_type == 'i':
            x_data = np.random.randint(-100, 100, size=size)
        elif data_type == 'f':
            x_data = np.random.uniform(-100, 100, size=size)
        elif data_type == 'p':
            x_data = np.random.uniform(0.0, 1.0, size=size)
        elif data_type == 'f+':
            x_data = np.random.uniform(0.0, 100.0, size=size)
            np.place(x_data, x_data == 0.0, 0.1)
        elif data_type == '0f+':
            x_data = np.random.uniform(0.0, 100.0, size=size)
        elif data_type == 'i+':
            x_data = np.random.randint(1, 100, size=size)
        elif data_type == 'b':
            x_data = np.random.randint(2, size=size)
        elif data_type == '(0,1)':
            arr = np.random.sample(size)
            np.place(arr, arr == 0.0, 0.1)
            x_data = arr
        elif data_type == '0i+':
            x_data = np.random.randint(0, 100, size=size, dtype=np.int)
        else:
            NotImplementedError('Unsupported type ' + str(data_type))
    return x_data


def get_new_var_name(prefix=''):
    global varCount
    varCount += 1
    if len(prefix) == 0:
        prefix = 'p'
    return prefix + str(varCount)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_limit(x):
    y = sigmoid(x)
    y = np.where(y == 0.0, np.finfo(np.float32).eps, y)
    y = np.where(y == 1.0, 1.0 - np.finfo(np.float32).eps, y)
    return y


def cast_data(y, output_type):
    if output_type == 'i':
        y = y.astype(np.int32)
    elif output_type == 'f':
        y = y.astype(np.float32)
    elif output_type == 'p':
        y = sigmoid(y)
    elif output_type == '(0,1)':
        y = sigmoid_limit(y)
    elif output_type == 'f+':
        y = np.abs(y)
        y = np.where(y == 0, np.finfo(np.float32).eps, y)
    elif output_type == '0f+':
        y = np.abs(y)
        y = y.astype(np.float32)
    elif output_type == 'i+':
        y = np.abs(y)
        y = np.where(y == 0, 1, y).astype(np.int32)
    elif output_type == '0i+':
        y = np.abs(y).astype(np.int32)
    elif output_type == 'b':
        y = np.round(sigmoid(y))
        y = y.astype(np.int32)
    return y


def generate_linear_data(size, output_type):
    x = np.random.uniform(0, 100, size=size)
    weight = np.random.randint(10)
    bias = np.random.randint(10)
    y = np.multiply(x, weight) + np.repeat(bias, size)
    y = cast_data(y, output_type)
    return x, y, weight, bias


def generate_linear_data2D(size, output_type):
    x = np.random.uniform(0, 100, size=size)
    weight = np.random.rand(size[1]) * 50  # produces something between 0 and 50
    bias = np.random.rand(1) * 20  # produces something between 0 and 20
    y = np.dot(x, weight) + bias
    # FIXME : determine whether data needs to be cast
    y = cast_data(y, output_type)

    return x, y, weight, bias


def parse_models(inference=False, funcs=False):
    with open('models.json') as modelFile:
        models = json.load(modelFile)
        # filter models or inferences if ignore flag is present
        filteredModels = []
        if inference:
            models = models["inferences"]
        elif funcs:
            models = models["util_functions"]
        else:
            models = models["models"]

        for m in models:
            if 'ig' not in m:
                filteredModels.append(m)
            elif not m['ig']:
                filteredModels.append(m)

    return filteredModels


def read_config():
    with open('config.json') as configFile:
        configs = json.load(configFile)
        return configs

def includes(candidate_support, support):
    if candidate_support == 'x':
        return True

    if support == 'f+':
        return candidate_support in ['f+', 'i+', '(0,1)']
    elif support == 'i+':
        return candidate_support in ['i+', '(0,1)']
    elif support == 'f':
        return candidate_support in ['f', 'f+', 'i+', 'i', '0i+', '(0,1)', 'p', '0f+', 'b']
    elif support == 'i':
        return candidate_support in ['i+', 'i', '0i+', '(0,1)']
    elif support == 'p':
        return candidate_support in ['(0,1)', 'p', 'b']
    elif support == '(0,1)':
        return candidate_support in ['(0,1)']
    elif support == '0i+':
        return candidate_support in ['0i+', '(0,1)', 'i+', 'p', 'b']
    elif support == '0f+':
        return candidate_support in ['f+', 'i+', '0i+', '(0,1)', 'p', '0f+']

    else:
        print('Unsupported type' + str(support))
        exit(-1)


def is_positive(type):
    return type in ['f+', 'i+', '(0,1)', '0i+', 'p', '0f+']

def isinteger(data):
    if type(data) is np.ndarray:
        return issubclass(data.dtype.type, np.integer)
    elif isinstance(data, list):
        return type(data[0]) is int
    else:
        return type(data) is int


def generateData(dim, type, is_special=False):
    x_data = generate_primitives(type, dim, is_special)
    return x_data


def isintegertype(type):
    return type in ['i', 'i+', '0i+', 'b']