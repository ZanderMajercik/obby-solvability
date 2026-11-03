import requests

base_url = "localhost:8080"


def test_request_all_levels():
    """ Test the request_all_levels endpoint """
    url = f"http://{base_url}/requestAllLevels"
    response = requests.get(url)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert "levels" in data, "Response JSON does not contain 'levels'"
    assert isinstance(data["levels"], list), "'levels' should be a list"
    assert len(data["levels"]) > 0, "'levels' list should not be empty"
    print("Test request_all_levels passed!")


def test_request_level_type():
    """ Test the request_level_type endpoint """
    level_type = "LavaSpinner"
    url = f"http://{base_url}/requestLevelType/{level_type}"
    response = requests.get(url)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    data = response.json()
    assert "parameters" in data, "Response JSON does not contain 'parameters'"
    assert isinstance(data["parameters"], list), "'parameters' should be a list"
    assert len(data["parameters"]) > 0, "'parameters' list should not be empty"
    print("Test request_level_type passed!")

def send_current_parameters():
    """ Test the send_current_parameters endpoint with the level type's default parameters """
    level_type = "LavaSpinner"
    parameters = requests.get(f"http://{base_url}/requestLevelType/{level_type}").json()["parameters"]
    url = f"http://{base_url}/sendCurrentParameters"
    response = requests.post(url, json={"level_type": level_type, "parameters": parameters})
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    assert "level_json" in response.json(), "Response JSON does not contain 'level_json'"
    print("Test send_current_parameters passed!")


def run_all_tests():
    """ Run all tests """
    test_request_all_levels()
    test_request_level_type()
    send_current_parameters()


if __name__ == "__main__":
    run_all_tests()
