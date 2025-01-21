import threading
import time

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
        self.lock = threading.Lock()  # To prevent race conditions

    def withdraw(self, amount):
        with self.lock:  # Acquire the lock
            if self.balance >= amount:
                print(f"{threading.current_thread().name} is withdrawing ${amount}")
                time.sleep(1)  # Simulate time for transaction
                self.balance -= amount
                print(f"{threading.current_thread().name} completed the withdrawal. Remaining balance: ${self.balance}")
            else:
                print(f"{threading.current_thread().name} tried to withdraw ${amount}, but insufficient balance!")

def customer_actions(account, amount):
    account.withdraw(amount)

# Main function
if __name__ == "__main__":
    # Create a shared bank account with $500
    account = BankAccount(500)

    # Create threads for customers
    customers = []
    for i in range(5):
        customer_thread = threading.Thread(target=customer_actions, args=(account, 200), name=f"Customer-{i+1}")
        customers.append(customer_thread)

    # Start all threads
    for thread in customers:
        thread.start()

    # Wait for all threads to complete
    for thread in customers:
        thread.join()

    print("All transactions are complete.")
