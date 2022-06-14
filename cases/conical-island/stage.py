def main():
    print("Preparing stage file...")
    
    with open("conical-island.stage", 'w') as fp:
        stages = (
            "4\n" +
            " 9.36 13.80\n" +
            "10.36 13.80\n" +
            "12.96 11.22\n" +
            "15.56 13.80"
        )
        
        fp.write(stages)

if __name__ == "__main__":
    main()