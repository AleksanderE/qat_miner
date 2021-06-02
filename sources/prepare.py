def _replace_eot(text):
    return text\
        .replace('<EOT>', '[unused99]')\
        .replace('[EOT]', '[unused99]')\
        .replace('<eot>', '[unused99]')\
        .replace('[eot]', '[unused99]')


def _replace_sep(text):
    return text\
        .replace('<SEP>', '[SEP]')\
        .replace('<sep>', '[SEP]')\
        .replace('[sep]', '[SEP]')


def _replace_code(text):
    return text\
        .replace('<CODE>', '[unused98]')\
        .replace('<code>', '[unused98]')


def _replace_join(text):
    return text\
        .replace('<JOIN>', '[unused97]')\
        .replace('[JOIN]', '[unused97]')\
        .replace('<join>', '[unused97]')\
        .replace('[join]', '[unused97]')


def _replace_speakers(text):
    for i in range(1, 97):
        text = text.replace('<SPEAKER' + str(i) + '>', '[unused' + str(i) + ']')
        text = text.replace('[SPEAKER' + str(i) + ']', '[unused' + str(i) + ']')
        text = text.replace('<speaker' + str(i) + '>', '[unused' + str(i) + ']')
        text = text.replace('[speaker' + str(i) + ']', '[unused' + str(i) + ']')
    return text


def prepare_special_tokens(text):
    if type(text) != str:
        return ''
    text = _replace_eot(text)
    text = _replace_sep(text)
    text = _replace_code(text)
    text = _replace_speakers(text)
    text = _replace_join(text)
    return text


def join_sep(text1, text2):
    return text1 + ' [SEP] ' + text2


def get_speakers_number(df):
    for i in range(1, 50):
        speaker_template = '[unused{}]'.format(i)
        found = False
        for _, row in df.iterrows():
            if speaker_template in row['message1'] or speaker_template in row['message2']:
                found = True
                break
        if not found:
            return i - 1
    assert False


def verify_dataset(df):
    for _, row in df.iterrows():
        assert '[EOT]' not in row['message1']
        assert '<EOT>' not in row['message1']
        assert '<SEP>' not in row['message1']
        assert '<SPEAKER' not in row['message1']
        assert '[SPEAKER' not in row['message1']

        assert '[eot]' not in row['message1']
        assert '<eot>' not in row['message1']
        assert '<sep>' not in row['message1']
        assert '<speaker' not in row['message1']
        assert '[speaker' not in row['message1']

        assert '[EOT]' not in row['message2']
        assert '<EOT>' not in row['message2']
        assert '<SEP>' not in row['message2']
        assert '<SPEAKER' not in row['message2']
        assert '[SPEAKER' not in row['message2']

        assert '[eot]' not in row['message2']
        assert '<eot>' not in row['message2']
        assert '<sep>' not in row['message2']
        assert '<speaker' not in row['message2']
        assert '[speaker' not in row['message2']
        
        if 'new_text_with_SEP_tag' in df.columns:
            assert '[EOT]' not in row['new_text_with_SEP_tag']
            assert '<EOT>' not in row['new_text_with_SEP_tag']
            assert '<SEP>' not in row['new_text_with_SEP_tag']
            assert '<SPEAKER' not in row['new_text_with_SEP_tag']
            assert '[SPEAKER' not in row['new_text_with_SEP_tag']
        
            assert '[eot]' not in row['new_text_with_SEP_tag']
            assert '<eot>' not in row['new_text_with_SEP_tag']
            assert '<sep>' not in row['new_text_with_SEP_tag']
            assert '<speaker' not in row['new_text_with_SEP_tag']
            assert '[speaker' not in row['new_text_with_SEP_tag']
        
        if 'pair_concatenated' in df.columns:
            assert '[EOT]' not in row['pair_concatenated']
            assert '<EOT>' not in row['pair_concatenated']
            assert '<SEP>' not in row['pair_concatenated']
            assert '<SPEAKER' not in row['pair_concatenated']
            assert '[SPEAKER' not in row['pair_concatenated']
            
            assert '[eot]' not in row['pair_concatenated']
            assert '<eot>' not in row['pair_concatenated']
            assert '<sep>' not in row['pair_concatenated']
            assert '<speaker' not in row['pair_concatenated']
            assert '[speaker' not in row['pair_concatenated']
