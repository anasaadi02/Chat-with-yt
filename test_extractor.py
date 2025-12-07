"""
Simple test script for YouTube extractor
"""

from youtube_extractor import extract_youtube_data, display_results

# Test with a popular TED talk
test_url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"

print("Testing YouTube Extractor...")
print("Test URL:", test_url)
print()

try:
    data = extract_youtube_data(test_url)
    display_results(data)
    print("\nTest completed successfully!")
except Exception as e:
    print("\nTest failed:")
    print(str(e))
    import traceback
    traceback.print_exc()
