import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from src.memory.db import get_session, EntityNode, GraphEdge


def get_all_data():
    with get_session() as session:
        nodes = session.query(EntityNode).all()
        edges = session.query(GraphEdge).all()

        result_data = []
        node_map = {node.id: node for node in nodes}

        for edge in edges:
            source = node_map.get(edge.source_id)
            target = node_map.get(edge.target_id)
            if source and target:
                result_data.append({
                    'start': source.name,
                    'end': target.name,
                    'r': [None, edge.relation]
                })
        return result_data


def visualize_graph(data=None):
    if data is None:
        data = get_all_data()

    G = nx.DiGraph()  # Directed graph

    for record in data:
        start = record['start']
        end = record['end']
        relationship = record['r'][1]
        G.add_node(start)
        G.add_node(end)
        G.add_edge(start, end, relationship=relationship)

    # 🧠 Use Kamada-Kawai layout with large scale for spacing
    pos = nx.spring_layout(G, k=3.0, iterations=100)  # Scale up to push nodes apart

    # 📺 Large full-screen figure
    fig = plt.figure(figsize=(16, 16))
    if hasattr(fig.canvas.manager, 'full_screen_toggle'):
        fig.canvas.manager.full_screen_toggle()

    # 🟦 Draw graph
    nx.draw(
        G, pos, with_labels=True,
        node_size=1500, node_color='skyblue',
        font_size=5,
        edge_color='gray', ax=plt.gca()
    )

    # 🏷️ Edge labels — move further along edge to avoid overlap
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels,
        font_size=4, label_pos=0.7  # Shift away from node center
    )

    plt.title("Graph Visualization")
    plt.tight_layout()
    plt.show()


def visualize_interactive(data=None):
    net = Network(notebook=False, height="1000px", width="100%", bgcolor="#ffffff", font_color="black",
                  cdn_resources='remote', directed=True)
    net.force_atlas_2based(gravity=-50)

    # If data is provided, use the old format for backward compatibility
    if data is not None:
        for record in data:
            start = record['start']
            end = record['end']
            rel = record['r'][1]
            net.add_node(start, label=start)
            net.add_node(end, label=end)
            net.add_edge(start, end, label=rel)
    else:
        # Otherwise, fetch all available information from the database
        with get_session() as session:
            nodes = session.query(EntityNode).all()
            edges = session.query(GraphEdge).all()

            for node in nodes:
                # Add all available information to tooltip
                tooltip = (
                    f"<b>Name:</b> {node.name}<br>"
                    f"<b>Topic:</b> {node.topic}<br>"
                    f"<b>Tags:</b> {node.tags_json}<br>"
                    f"<b>Text:</b> {node.text}"
                )
                net.add_node(node.id, label=node.name, title=tooltip)

            for edge in edges:
                # Some edges might refer to missing nodes if referential integrity was off, handle safely
                net.add_edge(edge.source_id, edge.target_id, label=edge.relation, title=edge.relation)

    net.show("graph.html", notebook=False)


if __name__ == '__main__':
    print("Generating comprehensive interactive graph with all available database info...")
    visualize_interactive()
    print("Done. Saved to graph.html")
