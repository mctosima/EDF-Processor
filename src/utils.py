import os

def clear_terminal():
    command = 'cls' if os.name == 'nt' else 'clear'
    os.system(command)

def parse_input(user_input):
    segments = user_input.split(',')
    output = []
    for segment in segments:
        segment = segment.strip()  # Remove any leading/trailing whitespace
        if '-' in segment:
            start, end = segment.split('-')
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(segment))
    return output

def parse_input_str(user_input):
    segments = user_input.split(',')
    output = []
    for segment in segments:
        segment = segment.strip()  # Remove any leading/trailing whitespace
        if '-' in segment:
            start, end = map(int, segment.split('-'))
            output.extend(str(i) for i in range(start, end + 1))
        else:
            output.append(segment)
    return output
