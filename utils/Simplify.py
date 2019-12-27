from collections.abc import Mapping, Iterable, MutableSequence, Iterator, Collection


__main__ = [
    'method_iteration'
]


def method_iteration(
    methods, 
    params):
    """Run method(s) with multiple pairs of parameters.
    Generate hierarchical return tuple of given method(s) and parameter(s).

    Parameters
    ----------
    methods : method or list of method(s)
        Method(s) to run.
    
    params : dict or list of dict(s)
        Parameter(s) of input methods in form of {param_name: param_value}. \\
        Must note that : \\
            1. when methods is not list, input params must be dict; \\
            2. when methods are list, input params must be list of dict with same length; \\
            3. all parameter values are recommended in form of [value], [listed_params] 
            for listed parameters. \\
        To run method with multiple pairs of parameters, varying parameter values should be 
        in form of [value_1, value_2, ...]. \\
        Must note that : \\
            1. varying parameter values must be in same length for one method;
            2. one-item list parameter values are exceptions.

    Returns
    -------
    [
        [
            (returns)
            [, (returns)]  # hierarchy on param values
        ],
        [, [
            (return)
            [, (returns)]  # hierarchy on param values
        ]]# hierarchy on methods
    ]

    Usage
    -----
    method_iteration(
        methods=[method1, method2], 
        params=[
            {param1_1: value, param1_2: [value, value]}, 
            {param2_1: [value2_1, value2_2], param2_2: [value2_3, value2_4]}]
    )
    """
    if not callable(methods) and not isinstance(methods, Iterable):
        raise TypeError('method_iteration: methods must be callable or Iterable. ')

    if not isinstance(params, (Mapping, MutableSequence)):
        raise TypeError('method_iteration: params must be Mapping or MutableSequence. ')
    
    if isinstance(params, MutableSequence):
        if len(params) == 1:
            params = next(iter(params))
    if isinstance(params, MutableSequence):
        if len(params) == 1:
            params = next(iter(params))
    
    if isinstance(methods, Iterable) and isinstance(params, Mapping):
        raise TypeError('method_iteration: either methods or params is not MutableSequence. 1')
    if callable(methods) and isinstance(params, MutableSequence):
        raise TypeError('method_iteration: either methods or params is not MutableSequence. ')
    
    if isinstance(methods, Iterable) and isinstance(params, MutableSequence):
        print('methods: iterable; params: MutableSequence; ')
        
        methods = iter(methods)
        params  = iter(params)

        setinel = object()

        results = []

        while True:
            method = next(methods, setinel)
            param  = next(params, setinel)
            if setinel is method and setinel is not param or setinel is not method and setinel is param:
                raise ValueError('method_iteraion: methods and params does not match2. ')
            if setinel is method and setinel is param:
                break
            results.append(method_iteration(method, param))
            # print(' results: {}'.format(results))
        
        return results
    
    if callable(methods) and isinstance(params, Mapping):
        print('methods: callable; params: Mapping; ')
        setinel = object()
        
        has_list_params = False
        list_params_len = 0
        for key, value in params.items():

            if isinstance(value, MutableSequence):

                print(' listed params {}: len={}'.format(key, len(value)))

                if len(value) == 1:
                    params[key] = next(iter(value))
                    continue
                
                has_list_params = True
                if list_params_len != 0 and list_params_len != len(value):
                    raise ValueError('method_iteration: listed params of method does not match. ')
                list_params_len = len(value)
                params[key] = iter(value)
            else:
                params[key] = value

        if not has_list_params:
            return methods(**params)
        
        results = []
        while True:
            next_params = {}
            # print(' results:     {}'.format(results))         # for test
            for key, value in params.items():
                if isinstance(value, Iterator):
                    temp = next(value,setinel)
                    if setinel is temp:
                        return results
                    next_params[key] = temp
                else:
                    next_params[key] = value
            # print(' next_params: {}'.format(next_params))     # for test
            # if setinel in next_params.values():               # the setinel check is
            #     return results                                # no longer here for pd.df
            res = methods(**next_params)
            if res:
                results.append(res)

# TODO: type check, comment
def results_archive(
    results, 
    keys, 
    listed=True):
    """
    """
    if not isinstance(keys, MutableSequence):
        raise TypeError('results_archive: keys must be MutableSequence. ')
    listed_dict_res = []
    for one_method_key, one_method_res in zip(keys, results):
        if not one_method_key and not one_method_res:
            listed_dict_res.append(None)
            continue
        one_method_key = [one_method_key] if not isinstance(one_method_key, MutableSequence) else one_method_key
        dict_res = {}
        res_num = len(one_method_key)
        for key in one_method_key:
            dict_res[key] = []
        for one_param_pair_res in one_method_res:
            one_param_pair_res = [one_param_pair_res] if not isinstance(one_param_pair_res, Collection) else one_param_pair_res
            one_param_pair_res = list(one_param_pair_res)
            # print('res: {}'.format(one_param_pair_res))
            # print('key: {}'.format(one_method_key))
            for key, value in zip(one_method_key, one_param_pair_res):
                # print('dict_res[{}].append({})'.format(key, value))
                dict_res[key].append(value)
        listed_dict_res.append(dict_res)
    if listed:
        return listed_dict_res
    else:
        return tuple(listed_dict_res)