
if __name__ == '__main__': #original
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        adversarial_config = json.load(fp)

    model_path = adversarial_config["model_path"]
    with open(model_path+'/config.json') as fp:
        config = json.load(fp)

    config.update(adversarial_config)

    with open(model_path+'/w2i.json') as fp:
        w2i = json.load(fp)

    data = load_data(config=config, word_to_idx=w2i)

    model = torch.load(model_path+'/model')

    if config["cuda"]:
        model = model.cuda()


