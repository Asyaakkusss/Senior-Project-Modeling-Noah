import xml.etree.ElementTree as ET

def read_xml(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        with open("test_data.txt", "w") as file:
            print("Start translating data")
            file.write(f"Root tag: {root.tag}\n")
            file.write(f"Root attributes: {root.attrib}\n")

            step = 0

            for child in root:
                if child.tag == "Record" and ((child.get('sourceName') == "Noah’s Iphone") or (child.get('sourceName') == "Noah’s Apple Watch")):
                    step += 1
                    file.write(f"\n\nEntry number: {step}")
                    #file.write(f"\nTag: {child.tag}, Attributes: {child.attrib}\n")
                    file.write(f"\n\tSource: {child.get('sourceName')}")
                    file.write(f"\n\tType: {child.get('type')}")
                    file.write(f"\n\t{child.get('startDate')} - {child.get('endDate')}")
                    file.write(f"\n\tValue: {child.get('value')}{child.get('unit')}")
                
                    for sub_child in child:
                        file.write(f"\n\t\tSub-tag: {sub_child.tag}, Sub-attributes: {sub_child.attrib}")

            print("Finished translating data")
            
    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")

if __name__ == "__main__":
    file_path = '/Users/noahh/export.xml'
    read_xml(file_path)