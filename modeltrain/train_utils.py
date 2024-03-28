from imports import *



def make_pick_from_sides_dataset(missmatch, groupingd):
    usescores = False
    useonlyoneoracle = False
    useonlyonelayer = False
    truncate_last = 1

    def f(actions, scores, minmaxs):
        if usescores:
            x = grouppfsscores(actions, np.cumsum(scores, 1), groupingd, truncate_last)
        else:
            minmaxs = np.copy(minmaxs)
            if useonlyonelayer:
                minmaxs[:, 1:, :, :] = 0
            if useonlyoneoracle:
                minmaxs[:, :, :, 1:] = 0
            sort(actions, minmaxs)
            x = group(actions, minmaxs, groupingd)
        return (x,
                minmaxs[:, 0:1, 0, 0])

    pathd = "./data/pick_from_sides_data/d"
    pathl = "./data/pick_from_sides_data/l"
    initial_run = False
    leng = 8
    num_weaker_oracles = 2
    if initial_run:
        makepfs(leng, 5, num_weaker_oracles, truncate_last, pathd, pathl)
    lengtruncated = leng - truncate_last
    coord_channels = lengtruncated - groupingd
    return FileDatasetMinmax(pathd, pathl, f, coord_channels, missmatch)

def print_dataset_summary(dataset, coord_channels):
    print([np.unique(dataset.minmaxs[:, x]).tolist() for x in range(dataset.minmaxs.shape[1])])
    print(np.mean(np.var(dataset.minmaxs, 2), 0))
    print(dataset.data.shape, coord_channels, dataset.data[0][:, 0], dataset.data[0][:coord_channels, 0],
          dataset.data[0][coord_channels:, 0])
    print(dataset.data.shape, coord_channels, dataset.data[1][:, 0], dataset.data[1][:coord_channels, 0],
          dataset.data[1][coord_channels:, 0])
    print(dataset.data.shape, coord_channels, dataset.data[2][:, 0], dataset.data[2][:coord_channels, 0],
          dataset.data[2][coord_channels:, 0])


def badvalindsf(dataset, val_inds):
    bad_val_inds_separate, bad_cat_names = select_bad_indices_homeguard_seperate(dataset.maps[val_inds],
                                                                                 dataset.minmaxs[val_inds],
                                                                                 dataset.scores)

    bad_val_inds = np.unique(mysum(bad_val_inds_separate))
    print(bad_val_inds.shape[0], bad_val_inds.shape[0] / len(val_inds))
    bad_val_inds_separate = [np.unique(x) for x in bad_val_inds_separate]
    bad_val_inds = (bad_val_inds, bad_val_inds_separate, bad_cat_names)
    return bad_val_inds


def r(hs, dos, lr, lrd, rs, rshalf, rsquarter, ed, sharpness_centering, dataset, config):
    total_channels = dataset.data.shape[1]
    coord_channels = dataset.coord_channels
    perc_validation = 0.1
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1 - perc_validation, perc_validation])
    val_inds = val_dataset.indices
    maps = dataset.map_strings[val_inds]

    def mapsf(x):
        for i in x:
            log(str(maps[i]) + "\n\n", "logs/maps.txt")
        log("\n\n", "logs/maps.txt")

    bad_val_inds = badvalindsf(dataset, val_inds)
    ms = dataset.minmaxs
    if len(ms.shape) > 3:
        ms = ms[:, :, :, 0]
    if config["reference_embeddings"]:
        ms = torch.from_numpy(ms[val_inds])
        meanminmax = torch.max(ms, 2)[0]
        auc_minmax_as_embedding = averageauc(torch.max(ms[:, 0:1], 2)[0], bad_val_inds, mapsf,
                                             fig_path="figs/minmax")
        print(f"auc_minmax_as_embedding = {auc_minmax_as_embedding}")
        auc_avg_minmax_as_embedding = averageauc(meanminmax, bad_val_inds, mapsf,
                                                 fig_path="figs/avg_minmax")
        print(f"auc_avg_minmax_as_embedding = {auc_avg_minmax_as_embedding}")
        auc_random_embedding = averageauc(torch.rand([len(val_inds), ed]), bad_val_inds, mapsf, fig_path="figs/random")
        print(f"auc_random_embedding = {auc_random_embedding}")
    config["learning_rate"] = lr
    config["learning_rate_decay"] = lrd
    config["embedding_dim"] = ed
    config["sharpness"] = sharpness_centering[0]
    config["center_momentum"] = sharpness_centering[1]

    def make_backbone():
        return pointnet.TransformerEmbedder(total_channels, ed, hs, dos[0])

    backbone1 = make_backbone()
    backbone2 = make_backbone()
    autoEncoder = pointnet.TransformerAutoEncoder(backbone1, coord_channels,
                                                  total_channels - coord_channels, dos[1])

    def make_dino(backbone):
        if config['contrastive_after'] >= config['max_epochs'] or config['perc_contrastive'] <= 0:
            return nn.Identity()
        return pointnet.PointNetDino(backbone, (hs[0], hs[0]), ed)

    dino = make_dino(backbone1)
    teacher = make_dino(backbone2)
    model = pointnet.DinoCompletionModuleList(autoEncoder,
                                              dino, teacher).to(config["dtype"])
    # model = pointnet.PointNetAutoEncoder(total_channels, coord_channels, total_channels, 16, hs)
    loss = train.main(model, train_dataset, val_dataset, t_dino.train, bad_val_inds, config)
    if "visualize" in config:
        model.eval()
        with torch.no_grad():
            final_scores = dataset.minmaxs[:, 0:1, 0][val_inds]
            x = torch.stack([val_dataset[i][0] for i in range(100)], 0)
            xyz = model.embedding(x.cuda()).cpu().numpy()
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=final_scores)
            plt.show()
    model.eval()
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size']
        , shuffle=False)
    embeds = []
    embedshalf = []
    embedquarter = []
    for batch_val in val_dataloader:
        data = batch_val[0].to(config["device"])
        with torch.no_grad():
            embeds.append(model.embedding(data))
            l = data.shape[2]
            shuffleinds = np.arange(l)
            np.random.shuffle(shuffleinds)
            embedshalf.append(model.embedding(data[:, :, shuffleinds[:l // 2]]))
            embedquarter.append(model.embedding(data[:, :, shuffleinds[:l // 4]]))
    embeds = torch.concatenate(embeds, 0)
    embedshalf = torch.concatenate(embedshalf, 0)
    embedquarter = torch.concatenate(embedquarter, 0)
    print("--full--")
    print_info(embeds, bad_val_inds, loss, (lr, lrd, hs, dos), rs, mapsf, True)
    #print("--half--")
    #print_info(embedshalf, bad_val_inds, loss, (lr, lrd, hs, dos), rshalf, mapsf, False)
    #print("--quarter--")
    #print_info(embedquarter, bad_val_inds, loss, (lr, lrd, hs, dos), rsquarter, mapsf, False)
