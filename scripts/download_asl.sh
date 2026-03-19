#!/bin/bash
# Download ASL Fingerspelling dataset files from Kaggle
# Usage: bash download_asl.sh [download_dir]

COMPETITION="asl-fingerspelling"
TEST_DATASET="sohier/529505295052950"
DOWNLOAD_DIR="${1:-./asl-fingerspelling}"

mkdir -p "$DOWNLOAD_DIR/supplemental_landmarks"
mkdir -p "$DOWNLOAD_DIR/train_landmarks"
mkdir -p "$DOWNLOAD_DIR/test_landmarks"

FILES_COMPETITION=(
    # Supplemental landmarks
    "supplemental_landmarks/371169664.parquet"
    "supplemental_landmarks/369584223.parquet"
    "supplemental_landmarks/1682915129.parquet"
    "supplemental_landmarks/775880548.parquet"
    "supplemental_landmarks/2100073719.parquet"
    "supplemental_landmarks/1650637630.parquet"
    "supplemental_landmarks/1471096258.parquet"
    "supplemental_landmarks/86446671.parquet"
    "supplemental_landmarks/897287709.parquet"
    "supplemental_landmarks/333606065.parquet"
    "supplemental_landmarks/2057261717.parquet"
    "supplemental_landmarks/971104021.parquet"
    "supplemental_landmarks/471766624.parquet"
    "supplemental_landmarks/1881515495.parquet"
    "supplemental_landmarks/1857374937.parquet"
    "supplemental_landmarks/293101677.parquet"
    "supplemental_landmarks/595441814.parquet"
    "supplemental_landmarks/1279694894.parquet"
    "supplemental_landmarks/756566775.parquet"
    "supplemental_landmarks/1471341722.parquet"
    "supplemental_landmarks/1112747136.parquet"
    "supplemental_landmarks/1756773911.parquet"
    "supplemental_landmarks/33432165.parquet"
    "supplemental_landmarks/1779786322.parquet"
    "supplemental_landmarks/1755047076.parquet"
    "supplemental_landmarks/1624527344.parquet"
    "supplemental_landmarks/597469033.parquet"
    "supplemental_landmarks/1505488209.parquet"
    "supplemental_landmarks/1144115867.parquet"
    "supplemental_landmarks/1249944812.parquet"
    "supplemental_landmarks/1118603411.parquet"
    "supplemental_landmarks/676340265.parquet"
    "supplemental_landmarks/716508881.parquet"
    "supplemental_landmarks/736978972.parquet"
    "supplemental_landmarks/1579345709.parquet"
    "supplemental_landmarks/285528514.parquet"
    "supplemental_landmarks/1047404576.parquet"
    "supplemental_landmarks/697480828.parquet"
    "supplemental_landmarks/1032110484.parquet"
    "supplemental_landmarks/440362090.parquet"
    "supplemental_landmarks/924144755.parquet"
    "supplemental_landmarks/639454452.parquet"
    "supplemental_landmarks/236903981.parquet"
    "supplemental_landmarks/636900267.parquet"
    "supplemental_landmarks/1176508147.parquet"
    "supplemental_landmarks/131312512.parquet"
    "supplemental_landmarks/778903889.parquet"
    "supplemental_landmarks/193950599.parquet"
    "supplemental_landmarks/442061898.parquet"
    "supplemental_landmarks/95345213.parquet"
    "supplemental_landmarks/1407656790.parquet"
    "supplemental_landmarks/680303484.parquet"
    "supplemental_landmarks/1727438550.parquet"
    # Train landmarks
    "train_landmarks/1358493307.parquet"
    "train_landmarks/495378749.parquet"
    "train_landmarks/2118949241.parquet"
    "train_landmarks/5414471.parquet"
    "train_landmarks/1133664520.parquet"
    "train_landmarks/433948159.parquet"
    "train_landmarks/1920330615.parquet"
    "train_landmarks/683666742.parquet"
    "train_landmarks/1365772051.parquet"
    "train_landmarks/939623093.parquet"
    "train_landmarks/1405046009.parquet"
    "train_landmarks/450474571.parquet"
    "train_landmarks/149822653.parquet"
    "train_landmarks/152029243.parquet"
    "train_landmarks/1552432300.parquet"
    "train_landmarks/1365275733.parquet"
    "train_landmarks/1880177496.parquet"
    "train_landmarks/1021040628.parquet"
    "train_landmarks/1557244878.parquet"
    "train_landmarks/1497621680.parquet"
    "train_landmarks/522550314.parquet"
    "train_landmarks/649779897.parquet"
    "train_landmarks/1905462118.parquet"
    "train_landmarks/175396851.parquet"
    "train_landmarks/638508439.parquet"
    "train_landmarks/532011803.parquet"
    "train_landmarks/2072296290.parquet"
    "train_landmarks/1906357076.parquet"
    "train_landmarks/2026717426.parquet"
    "train_landmarks/1967755728.parquet"
    "train_landmarks/1785039512.parquet"
    "train_landmarks/1643479812.parquet"
    "train_landmarks/1134756332.parquet"
    "train_landmarks/1019715464.parquet"
    "train_landmarks/566963657.parquet"
    "train_landmarks/568753759.parquet"
    "train_landmarks/1726141437.parquet"
    "train_landmarks/296317215.parquet"
    "train_landmarks/234418913.parquet"
    "train_landmarks/614661748.parquet"
    "train_landmarks/654436541.parquet"
    "train_landmarks/474255203.parquet"
    "train_landmarks/1662742697.parquet"
    "train_landmarks/1099408314.parquet"
    "train_landmarks/1341528257.parquet"
    "train_landmarks/105143404.parquet"
    "train_landmarks/527708222.parquet"
    "train_landmarks/882979387.parquet"
    "train_landmarks/933868835.parquet"
    "train_landmarks/1969985709.parquet"
    "train_landmarks/425182931.parquet"
    "train_landmarks/1098899348.parquet"
    "train_landmarks/1255240050.parquet"
    "train_landmarks/1997878546.parquet"
    "train_landmarks/128822441.parquet"
    "train_landmarks/388576474.parquet"
    "train_landmarks/546816846.parquet"
    "train_landmarks/1320204318.parquet"
    "train_landmarks/1448136004.parquet"
    "train_landmarks/349393104.parquet"
    "train_landmarks/2072876091.parquet"
    "train_landmarks/871280215.parquet"
    "train_landmarks/1562234637.parquet"
    "train_landmarks/1865557033.parquet"
    "train_landmarks/1664666588.parquet"
    "train_landmarks/1647220008.parquet"
    "train_landmarks/2036580525.parquet"
    "train_landmarks/169560558.parquet"
)

FILES_TEST=(
    "test_landmarks/111123288.parquet"
    "test_landmarks/1350482956.parquet"
    "test_landmarks/1620362322.parquet"
    "test_landmarks/265499232.parquet"
    "test_landmarks/714499365.parquet"
    "test_landmarks/1262950633.parquet"
    "test_landmarks/1363588998.parquet"
    "test_landmarks/1972425739.parquet"
    "test_landmarks/426444153.parquet"
    "test_landmarks/728455412.parquet"
    "test_landmarks/1291412689.parquet"
    "test_landmarks/1364634807.parquet"
    "test_landmarks/2078023931.parquet"
    "test_landmarks/455775941.parquet"
    "test_landmarks/788570001.parquet"
    "test_landmarks/1293700879.parquet"
    "test_landmarks/1502077106.parquet"
    "test_landmarks/2092171454.parquet"
    "test_landmarks/530315993.parquet"
    "test_landmarks/994619209.parquet"
    "test_landmarks/1313321259.parquet"
    "test_landmarks/1608371337.parquet"
    "test_landmarks/230007240.parquet"
    "test_landmarks/639657337.parquet"
)

TOTAL_COMPETITION=${#FILES_COMPETITION[@]}
TOTAL_TEST=${#FILES_TEST[@]}
TOTAL_ALL=$((TOTAL_COMPETITION + TOTAL_TEST))
echo "Competition files: $TOTAL_COMPETITION"
echo "Test files: $TOTAL_TEST"
echo "Total files (all sets): $TOTAL_ALL"

echo "Downloading $TOTAL files to $DOWNLOAD_DIR..."
echo "============================================="

for FILE in "${FILES_COMPETITION[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL_ALL] Downloading $FILE..."

    kaggle competitions download -c "$COMPETITION" -f "$FILE" -p "$DOWNLOAD_DIR/$(dirname "$FILE")" --force -q

    if [ $? -eq 0 ]; then
        # Unzip if downloaded as zip
        BASENAME=$(basename "$FILE")
        ZIPFILE="$DOWNLOAD_DIR/$(dirname "$FILE")/${BASENAME}.zip"
        if [ -f "$ZIPFILE" ]; then
            unzip -o -q "$ZIPFILE" -d "$DOWNLOAD_DIR/$(dirname "$FILE")"
            rm "$ZIPFILE"
        fi
        echo "  ✓ Done"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed"
    fi
done

for FILE in "${FILES_TEST[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL_ALL] Downloading $FILE..."

    #!/bin/bash
    kaggle datasets download $TEST_DATASET -f "$FILE" -p "$DOWNLOAD_DIR/$(dirname "$FILE")" --force -q

    if [ $? -eq 0 ]; then
        # Unzip if downloaded as zip
        BASENAME=$(basename "$FILE")
        ZIPFILE="$DOWNLOAD_DIR/$(dirname "$FILE")/${BASENAME}.zip"
        if [ -f "$ZIPFILE" ]; then
            unzip -o -q "$ZIPFILE" -d "$DOWNLOAD_DIR/$(dirname "$FILE")"
            rm "$ZIPFILE"
        fi
        echo "  ✓ Done"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ Failed"
    fi
done

echo "============================================="
echo "Complete: $((TOTAL_ALL - FAILED))/$TOTAL_ALL succeeded, $FAILED failed"
