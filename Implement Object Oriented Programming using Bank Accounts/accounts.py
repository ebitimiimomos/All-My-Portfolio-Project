
import random #import the random module 
import datetime #import the datetime module

#Define the BasicAccount class
class BasicAccount: 
      #Define the Class Variables
    acNum = 0

     #Using the dunder method to define the initialiser giving the account name and opening balance.
    def __init__(self, acName, openingBalance): #define the initialiser
        self.name = str(acName) #account holder name
        BasicAccount.acNum +=1
        self.acNum = BasicAccount.acNum #number of the account
        self.balance = float(openingBalance) #balance of the account

        
    # string representations that gives the account name and available balance
    def __str__(self):
        return f"{self.name}: £{self.balance}"

    #deposit stated ammount, adjust balance approprately ,deposit can only be positive
    def deposit(self, amount):
        if amount >= 0:
            self.balance += amount
        else: 
            print("error,please try again")

    #withdraw stated amount
    def withdraw(self, amount):
        if self.balance >= amount:
                self.balance -= amount
                print ("{self.name} has withdrawn £{amount}.New balance is £{self.balance}".format(self=self,amount=amount))
        else: #If an invalid amount is requested
            print ("\n Can not withdraw £{amount}".format(amount=amount))
    
    def getAvailableBalance(self): #Returns the total balance that is available in the account
        return self.balance
    
    def getBalance(self):
        return self.balance

    def printBalance(self): #the balance of the account
        print(f"{self.name} balance is £{self.balance}.")
        
    def getName(self): #Returns the name of the account holder
        return str(self.name)
    
    def getAcNum(self): #Returns the account number as a string
        return f"{self.acNum}"

         
    #Creates a new card number, with the expiry date being 3 years to the month from now
    def issueNewCard(self):
        self.cardNum = random.randint(1e15,9e15)
        today = datetime.date.today()
        self.cardExp = datetime.datetime.now()
        print("Your card expiry date is" ,self.cardExp)
        print("the month is" ,self.cardExp.month)        
        print("the year is" ,self.cardExp.year)
        self.cardExp = today. month, int(str(today.year+3)[2:])
        print ("Your new card number is", self.cardNum,self.cardExp)
    
    def closeAccount(self):#Close account before deleting of the object instance
        if self.balance >= 0:
            self.withdraw(self.balance) #Returns any balance to the customer 
            return True
        else: #if the customer is in debt to the bank
            print(f"This account cannot close because customer account overdrawn by £{(self.balance)}")
            return False


#Define the Premium Account Class
class PremiumAccount (BasicAccount):
    def __init__ (self, acName, openingBalance, initialOverdraft): #initialiser giving the account name, opening balance, and overdraft limit
        super().__init__(acName,openingBalance)
        self.overdraft = True
        self.overdraftLimit = float(initialOverdraft)

    def __str__(self):
        return f"Account holder name: {self.name} \nAccount balance £{self.balance} \nOverdraft Limit £{self.overdraftLimit}"

    #Sets the overdraft limit to the stated amount
    def setOverdraftLimit(self, max):
        self.overdraftLimit = max

    def withdraw(self, amount): 
        if self.balance + self.overdraftLimit >= amount: #if amount is larger than the balance and overdraft limit 
            self.balance -= amount
            print("{self.name} has withdrawn £{amount}. New balance is {self.balance}".format(self=self, amount=amount))
        else:
            print("Can not withdraw £{amount}".format(amount=amount))
            
    def getAvailableBalance(self):#Returns the total balance that is available in the account including overdraft
        return float(self.balance + self.overdraftLimit)
        
    def printBalance(self):
        if self.balance > 0:
            print("Hello {self.name}, your available balance is £{self.balance}, overdraft remaining is £{self.overdraftLimit}.".format(self=self))
        else:     
            self.overdraft = self.overdraftLimit + self.balance
            print("Hello {self.name}, your available balance is £{self.balance}, overdraft remaining is £{self.overdraft}.".format(self=self))
    
    def closeAccount(self):
        return super().closeAccount()
