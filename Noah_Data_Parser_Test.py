import xml.etree.ElementTree as ET

def read_xml(file_path):
    try:
        step = 0
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Print out the root tag and its attributes (if any)
        print(f"Root tag: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        # Iterate over each child element of the root
        for child in root:
            step += 1
            print(f"\nTag: {child.tag}, Attributes: {child.attrib}")
            # If the child has text, print it
            if child.text:
                print(f"Text: {child.text.strip()}")

            # Iterate over sub-elements of each child
            for sub_child in child:
                print(f"  Sub-tag: {sub_child.tag}, Sub-attributes: {sub_child.attrib}")
                if sub_child.text:
                    print(f"  Sub-text: {sub_child.text.strip()}")
            if step == 20:
                break

    except ET.ParseError as e:
        print(f"Failed to parse XML: {e}")
    except FileNotFoundError:
        print("File not found. Please check the file path.")

if __name__ == "__main__":
    # Replace 'your_file.xml' with your XML file path
    file_path = 'your_file.xml'
    read_xml(file_path)
