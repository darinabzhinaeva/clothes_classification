import os
import  pandas as pd
import json

pd.options.display.max_columns = 50

label_dict = {
    "AGE": {
        "type": "check_boxes",
        "values": {0: '0_2', 1: '2_8', 2: '9_14', 3: '15_24', 4: '25_34', 5: '35_44', 6: '45_54', 7: '55+'},
    },

    "SHAPE": {
        "type": "check_boxes",
        "values": {0: 'Column', 1: 'Hour-glass', 2: 'Pear', 3: 'Apple', 4: 'Full hour-glass', 5: 'Full pear'}
    },

    "OCCASION": {
        "type": "check_boxes",
        "values": {0: 'Casual', 1: 'Smart Office Attire', 2: 'Tea-party / Summer party / Day-time occasion',
                   3: 'Black tie', 4: 'Smart-Casual / Creative', 5: 'Beach', 6: 'Sport'}
    },

    "SEASON": {
        "type": "check_boxes",
        "values": {0: "summer", 1: "autumn", 2: "winter", 3: "spring"}
    },

    "PATTERN": {
        "type": "check_boxes",
        "values": {0: "yes"}
    },

    "SHADE": {
        "type": "radio_buttons",
        "values": {0: "light", 1: "true", 2: "dark", 3: "neon"}
    },

    "COLOR": {
        "type": "soft_select",
        "depend_on": "SHADE",
        "map": {
            "light": {0: "red", 1: "pink", 2: "green", 3: "grey", 4: "blue", 5: "turquoise", 6: "brown", 7: "white",
                      8: "black", 9: "beige", 10: "purple", 11: "violet", 12: "orange"},
            "true": {0: "red", 1: "pink", 2: "green", 3: "grey", 4: "blue", 5: "turquoise", 6: "brown", 7: "white",
                     8: "black", 9: "beige", 10: "purple", 11: "violet", 12: "orange"},
            "dark": {0: "red", 1: "pink", 2: "green", 3: "grey", 4: "blue", 5: "turquoise", 6: "brown", 7: "white",
                     8: "black", 9: "beige", 10: "purple", 11: "violet", 12: "orange"},
            "neon": {0: "red", 1: "pink", 2: "green", 4: "blue", 5: "turquoise", 10: "purple", 11: "violet",
                     12: "orange"}}
    },

    "GENDER": {
        "type": "radio_buttons",
        "values": {0: "male", 1: "female", 2: "unisex"}
    },

    "CATEGORIES": {
        "type": "radio_buttons",
        "values": {0: "Accessories", 1: "Tops", 2: "Trousers/Pants", 3: "Skirts", 4: "Dresses", 5: "Jeans",
                   6: "Jumpsuit", 7: "Sweaters/Jumpers", 8: "Cardigans", 9: "Bags", 10: "Shoes", 11: "Coats",
                   12: "Jackets", 13: "Blazers", 14: "Shorts", 15: "Underwear", 16: "Sarong", 17: "Coverups",
                   18: "Swimwear"}
    },

    "SUBCATEGORIES": {
        "type": "radio_buttons",
        "depend_on": "CATEGORIES",
        "map": {
            "Accessories": {0: "earrings", 1: "rings", 2: "sunglasses", 3: "necklaces", 4: "hair accessories",
                            5: "gloves", 6: "scarves", 7: "socks", 8: "tights", 9: "brooches", 10: "belts",
                            11: "bracelet"},
            "Tops": {0: "empty"},
            "Trousers/Pants": {0: "empty"},
            "Skirts": {0: "mini", 1: "midi", 2: "maxi"},
            "Dresses": {0: "mini", 1: "midi", 2: "maxi"},
            "Jeans": {0: "empty"},
            "Jumpsuit": {0: "empty"},
            "Sweaters/Jumpers": {0: "empty"},
            "Cardigans": {0: "empty"},
            "Bags": {0: "shoulder", 1: "clutch"},
            "Shoes": {0: "boots", 1: "pumps", 2: "flip-flops", 3: "sneakers", 4: "over-the-knee boots"},
            "Coats": {0: "empty"},
            "Jackets": {0: "empty"},
            "Blazers": {0: "empty"},
            "Shorts": {0: "empty"},
            "Underwear": {0: "bikini tops", 1: "bikini bottoms", 3: "one-piece"},
            "Sarong": {0: "empty"},
            "Coverups": {0: "empty"},
            "Swimwear": {0: "bikini tops", 1: "bikini bottoms", 3: "one-piece"}
        }
    },

    "MERCHANT_INFO": {
        "type": "line_edit",
        "values": ["Merchant", "Brand"]
    }
}

with open("folders_list.txt", "r") as f:
    file_text = f.read()

full_images_list = []
folders_list = file_text.split("\n")
for folder in folders_list:
    files_list = [os.path.join(folder, f) for f
                  in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    images_list = [x for x in files_list if not x.endswith("json")]
    full_images_list += images_list


data = {
    "PATH": [],
}

for key in label_dict:
    if label_dict[key]["type"] == "check_boxes":
        for value in label_dict[key]["values"].values():
            column_name = key + "_" + value
            data[column_name] = []

columns = list(data.keys())
columns.remove("PATH")

for image in full_images_list:
    json_file_path = image + ".json"
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as f:
            text = f.read()
        current_image_data = json.loads(text)
        data["PATH"].append(image)

        for key in columns:
            is_append = False
            for current_data_key in current_image_data.keys():
                if key.startswith(current_data_key):
                    values = current_image_data[current_data_key]
                    for value in values:
                        if key == current_data_key + "_" + value:
                            data[key].append(1)
                            is_append = True
            if not is_append:
                data[key].append(0)

df = pd.DataFrame(data)

df.to_csv("data.csv", sep=";", index=False)










