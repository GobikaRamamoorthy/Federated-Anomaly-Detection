import hashlib
import time

# Define the Block class
class Block:
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.timestamp = time.ctime()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        # Concatenate block contents and compute SHA-256 hash
        value = str(self.index) + self.timestamp + self.data + self.previous_hash
        return hashlib.sha256(value.encode()).hexdigest()

# Create the blockchain (a list of blocks)
genesis_block = Block(0, "Genesis Block", "0")
second_block = Block(1, "Vinithaa -> Sanjitha : 5 BTC", genesis_block.hash)
third_block = Block(2, "Sanjitha -> Priya : 2 BTC", second_block.hash)

blockchain = [genesis_block, second_block, third_block]

# Print the blockchain
print("\n=== Original Blockchain ===\n")
for block in blockchain:
    print(f"Block {block.index}")
    print(f"Timestamp     : {block.timestamp}")
    print(f"Data          : {block.data}")
    print(f"Prev Hash     : {block.previous_hash}")
    print(f"Current Hash  : {block.hash}\n")

# Tampering demonstration
print("=== Tampering Example ===")
second_block.data = "Vinithaa -> Sanjitha : 50 BTC"
tampered_hash = second_block.calculate_hash()
print(f"Modified Data : {second_block.data}")
print(f"New Hash      : {tampered_hash}")
print(f"Old Hash      : {second_block.hash}\n")

if tampered_hash != second_block.hash:
    print("⚠️  Tampering detected! The blockchain is no longer valid.")
else:
    print("✅  Blockchain is still valid.")
