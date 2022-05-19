def  string_to_point(string):
    characters = "() "
    for x in range(len(characters)):
        string = string.replace(characters[x], '')
    a = string.split(',')
    return (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (int(a[4]), int(a[5]))