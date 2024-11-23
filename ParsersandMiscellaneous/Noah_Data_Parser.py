#This file primarily serves as a way to view the data
import xml.etree.ElementTree as ET

def read_xml(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        print("Start translating data")
        print(f"Root tag: {root.tag}\n")
        print(f"Root attributes: {root.attrib}\n")
        data_files = {}
        step_counters = {}
        subs = {}

        for child in root:
            # Filter for Noah's health data
            if child.tag == "Record" and ((child.get('sourceName') == "Noah’s Iphone") or (child.get('sourceName') == "Noah’s Apple Watch")):
                record_type: str = child.get("type")

                char_cut_off = len("HKQuantityTypeIdentifier")
                # Open a new file for each type of data if not already opened
                if record_type not in data_files:
                    data_files[record_type] = open(f"{record_type[char_cut_off:]}_data.txt", "w")
                    step_counters[record_type] = 0

                file = data_files[record_type]
                step_counters[record_type] += 1


                # Write the data into the file
                file.write(f"\n\nEntry number: {step_counters[record_type]}")
                file.write(f"\n\tSource: {child.get('sourceName')}")
                file.write(f"\n\tType: {child.get('type')}")
                file.write(f"\n\t{child.get('startDate')} - {child.get('endDate')}")
                if child.get('unit'):
                    file.write(f"\n\tValue: {child.get('value')}{child.get('unit')}")
                else:
                    file.write(f"\n\tValue: {child.get('value')}")

                for sub_child in child:
                    file.write(f"\n\t\tSub-tag: {sub_child.tag}, Sub-attributes: {sub_child.attrib}")
                    if record_type not in subs:
                        subs[record_type] = sub_child.tag, sub_child.attrib

        print("Finished translating data")
        #print(data_files.keys())
        print(subs)

        # Close all the files after processing
        for file in data_files.values():
            file.close()

    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")

if __name__ == "__main__":
    file_path = '/Users/noahh/Downloads/apple_health_export 3/export.xml'
    read_xml(file_path)