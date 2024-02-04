
### CRC
CRC_removed_path = 'CRC_removed_feature_list.txt'
with open(CRC_removed_path, 'r') as file:
    CRC_removed_list = file.read().splitlines()
print("len(CRC_removed_list):", len(CRC_removed_list))

### Breast ST
breast_removed_path = 'breast_removed_feature_list.txt'
with open(breast_removed_path, 'r') as file:
    breast_removed_list = file.read().splitlines()
print("len(breast_removed_list):", len(breast_removed_list))

### cSCC
cSCC_removed_path = 'cSCC_removed_feature_list.txt'
with open(cSCC_removed_path, 'r') as file:
    cSCC_removed_list = file.read().splitlines()
print("len(cSCC_removed_list):", len(cSCC_removed_list))

# Union list
union_list = list(set(CRC_removed_list) | set(breast_removed_list) | set(cSCC_removed_list))
print("len(union_list):", len(union_list))
with open("removed_feature_union.txt", "w") as file:
    for item in union_list:
        file.write(item + "\n")
     
