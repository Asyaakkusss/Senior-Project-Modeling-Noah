import xml.etree.ElementTree as ET

def read_xml(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        with open("test_data.txt", "w") as file:
            print("Start translating data")
            # Print out the root tag and its attributes (if any)
            file.write(f"Root tag: {root.tag}")
            file.write(f"Root attributes: {root.attrib}")
            step = 0

            # Iterate over each child element of the root
            for child in root:
                step+=1
                file.write(f"\nTag: {child.tag}, Attributes: {child.attrib}")
                file.write(child.get("type"))
                file.write(f"{child.get("startDate")} - {child.get("endDate")}")
                file.write(f"{child.get("value")}{child.get("unit")}")

                if child.text:
                    file.write(f"Text: {child.text.strip()}")

                for sub_child in child:
                    print(f"  Sub-tag: {sub_child.tag}, Sub-attributes: {sub_child.attrib}")
                    if sub_child.text:
                        print(f"  Sub-text: {sub_child.text.strip()}")
                if step == 20:
                    print("Finished translating data")
                    break
            

    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")

if __name__ == "__main__":
    # Replace 'your_file.xml' with your XML file path
    file_path = 'export.xml'
    read_xml(file_path)