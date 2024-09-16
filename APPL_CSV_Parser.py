#This is a dedicated parser to convert the data to a csv file
import xml.etree.ElementTree as ET
import pandas as pd

def read_xml(file_path) -> dict:
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        print("Start translating data")
        print(f"Root tag: {root.tag}\n")
        print(f"Root attributes: {root.attrib}\n")
        health_data = {}

        for child in root:
            # Filter for Noah's health data
            if child.tag == "Record" and ((child.get('sourceName') == "Noah’s Iphone") or (child.get('sourceName') == "Noah’s Apple Watch")):
                record_type: str = child.get("type")

                if record_type.startswith("HKQuantityTypeIdentifier"):
                    cut_record_type = record_type[len("HKQuantityTypeIdentifier"):]
                else:
                    cut_record_type = record_type
                # Open a new file for each type of data if not already opened

                if cut_record_type not in health_data:
                    health_data[cut_record_type] = []

                duration = find_duration(child.get('startDate'), child.get('endDate'))

                #add health data into lists
                row_data = {
                        'source': child.get('sourceName'),
                        'start': child.get('startDate'),
                        'end': child.get('endDate'),
                        'duration': duration,
                        'value': child.get('value'),
                        'unit': child.get('unit'),
                        'sub_children': [sub_child.attrib for sub_child in child]
                    }

                health_data[cut_record_type].append(row_data)

        #convert lists to dataframes
        health_dataframes = {}
        for key, data_list in health_data.items():
            health_dataframes[key] = pd.DataFrame(data_list)

        print("Finished translating data")
        #print(data_files.keys())
        return health_dataframes

    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")

def find_duration(start, end):
    beginning = pd.to_datetime(start)
    finish = pd.to_datetime(end)
    return beginning - finish

def pd_to_csv(data_dict):
    for key in data_dict.keys():
        data_dict[key].to_csv(f"{key}.csv")

if __name__ == "__main__":
    file_path = '/Users/noahh/Downloads/apple_health_export 3/export.xml'
    health_data = read_xml(file_path)
    pd_to_csv(health_data)
    print("Finished translating data")
