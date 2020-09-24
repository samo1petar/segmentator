
def encode_phone_name(name: str) -> str:

    encode_names = {
        'high_range' : 'huawei_p20',
        'mid_range' : 'huawei_ale_l21',
        'low_range' : 'samsung_sm-g386f',
    }

    return encode_names[name]


def decode_phone_name(name: str) -> str:

    decode_names = {
        'huawei_p20'      : 'high_range',
        'huawei_ale_l21'  : 'mid_range',
        'samsung_sm-g386f': 'low_range',
    }

    return decode_names[name]
