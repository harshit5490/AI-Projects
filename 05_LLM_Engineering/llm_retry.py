import time


def retry_operation(operation, max_retries=3):

    for attempt in range(max_retries):

        try:
            return operation()

        except Exception as error:

            if attempt == max_retries - 1:
                raise

            delay = 2 ** attempt

            print(
                f"Request failed: {error}"
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)