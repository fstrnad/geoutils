import string


def remove_string_containing(string_list, substring):
    """
    Remove items from the list that contain the specified substring.

    Parameters:
    substring (str): The substring to check for in each item of the list.
    string_list (list of str): The list of strings to be filtered.

    Returns:
    list of str: A new list containing items from string_list that do not
                  contain the specified substring.

    Example:
    >>> my_list = ["apple", "banana", "cherry", "date", "grape"]
    >>> remove_items_containing("a", my_list)
    ['cherry']
    """
    return [item for item in string_list if substring not in item]


def find_string_containing(string_list, substring):
    """
    Find items from the list that contain the specified substring.

    Parameters:
    substring (str): The substring to check for in each item of the list.
    string_list (list of str): The list of strings to be filtered.

    Returns:
    list of str: A new list containing items from string_list that contain
                  the specified substring.

    Example:
    >>> my_list = ["apple", "banana", "cherry", "date", "grape"]
    >>> find_items_containing("a", my_list)
    ['apple', 'banana', 'date', 'grape']
    """
    return [item for item in string_list if substring in item]
