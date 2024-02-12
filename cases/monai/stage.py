if __name__ == "__main__":
    with open("monai.stage", 'w') as fp:
        print("Preparing stage file...")
        
        stages = (
            "3\n" +
            "4.521 1.196\n" +
            "4.521 1.696\n" +
            "4.521 2.196"
        )        
        
        fp.write(stages)