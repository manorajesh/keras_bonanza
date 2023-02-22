import csv

def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False
    sqr = int(n**0.5) + 1
    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True

def main():
    with open('primes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['num', 'is_prime'])
        for i in range(1, 100000):
            if is_prime(i):
                writer.writerow([i, 1])
            else:
                writer.writerow([i, 0])

if __name__ == '__main__':
    main()