def token_size(input_text):
    # A simple approximation: 1 token is roughly 4 characters.
    # This removes the need for Hugging Face or heavy downloads.
    if not input_text:
        return 0
    return len(input_text) // 4
