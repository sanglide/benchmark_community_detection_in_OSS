from infomap import Infomap


def custom_infomap(G):
    """
        输入无向赋权图G，输出infomap算法对G的社区发现结果
    """

    integer_to_string = {}
    string_to_integer = {}
    counter = 0

    def map_string_to_integer(string):
        """
            将string映射到自然数
        """
        nonlocal counter
        if string not in integer_to_string.values():
            string_to_integer[string] = counter
            integer_to_string[counter] = string
            counter += 1
        return string_to_integer[string]

    im = Infomap("--silent")
    for edge in G.edges(data=True):
        source = map_string_to_integer(edge[0])
        target = map_string_to_integer(edge[1])
        weight = edge[2]['weight']
        im.add_link(source, target, weight)
    for node in G.nodes(data=True):
        im.add_node(map_string_to_integer(node[0]))
    im.run()
    # print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
    im_communities = [[] for _ in range(im.num_top_modules)]
    max_id = 0
    for node in im.iterLeafNodes():
        if node.node_id > max_id:
            max_id = node.node_id
        im_communities[node.module_id - 1].append(integer_to_string[node.node_id])
    # print("im:", len(im_communities))
    return im_communities





