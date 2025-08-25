import requests
import json


def get_attestation_size():
    # url = "http://localhost:4100/eth/v1/beacon/blocks/head/attestations"
    url = "http://testing.mainnet.beacon-api.nimbus.team/eth/v1/beacon/blocks/head/attestations"
    headers = {"accept": "application/json"}
    
    # with open('intime/n_attestations.json', 'r') as file:
    #     data = json.load(file)
    
    try:
        # Send GET request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        
        # Parse JSON response
        data = response.json()

        # Get the list of attestations
        attestations = data.get('data', [])
        
        # Calculate the number of attestations
        attestation_count = len(attestations)
        
        if attestation_count == 0:
            print("error: No attestations found")
        
        # Calculate the average size of a single attestation
        total_size = sum(len(json.dumps(att)) for att in attestations)
        average_size = total_size / attestation_count
        
        print(f"attestation_count: {attestation_count}")
        print(f"average_attestation_size_bytes: {round(average_size, 2)}")
        print(f"average_attestation_size_kb: {round(average_size / 1024, 2)}")
            
    except requests.exceptions.RequestException as e:
        print(f"error Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"error JSON parsing failed: {str(e)}")

get_attestation_size()
