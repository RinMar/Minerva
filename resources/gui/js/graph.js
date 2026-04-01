// Use window.pyBridge if available, otherwise try window.parent.pyBridge
function getBridge() {
    return window.pyBridge || (window.parent && window.parent.pyBridge); 
}

// ── Graph Setup ──
var nodes = new vis.DataSet([]);
var edges = new vis.DataSet([]);
var network = null;

function initNetwork() {
    const container = document.getElementById('mynetwork');
    const data = { nodes: nodes, edges: edges };

    const options = {
        interaction: {
            dragNodes: true,
            hover: true,
            zoomView: true,
            dragView: false
        },
        physics: {
            enabled: true,
            stabilization: { iterations: 120 },
            barnesHut: {
                gravitationalConstant: -3000,
                springLength: 160,
                springConstant: 0.03,
                damping: 0.12
            }
        },
        nodes: {
            shape: 'dot',
            size: 18,
            font: { size: 13, color: '#ccc', face: 'Inter, sans-serif' },
            color: {
                background: '#7c3aed',
                border: '#6d28d9',
                highlight: { background: '#a78bfa', border: '#7c3aed' },
                hover: { background: '#8b5cf6', border: '#7c3aed' }
            },
            borderWidth: 2
        },
        edges: {
            color: { color: '#444', highlight: '#888', hover: '#666' },
            font: { size: 10, color: '#777', face: 'Inter, sans-serif', strokeWidth: 0 },
            arrows: { to: { enabled: true, scaleFactor: 0.6 } },
            smooth: { type: 'curvedCW', roundness: 0.15 }
        }
    };

    network = new vis.Network(container, data, options);
}

// ── Called from Parent (or Python if no iframe) ──
window.updateGraph = function(dataUpdate) {
    if (dataUpdate.nodes) {
        var newNodes = dataUpdate.nodes.map(function(n) { return String(n.id); });
        var currentNodes = nodes.getIds();
        var nodesToRemove = currentNodes.filter(function(id) { return !newNodes.includes(String(id)); });
        if (nodesToRemove.length > 0) {
            nodes.remove(nodesToRemove);
        }

        dataUpdate.nodes.forEach(function (n) {
            n.id = String(n.id);
            if (nodes.get(n.id)) {
                nodes.update(n);
            } else {
                nodes.add(n);
            }
        });
    }

    if (dataUpdate.edges) {
        var newEdges = [];
        dataUpdate.edges.forEach(function (e) {
            var id = e.id || (String(e.from) + "_" + String(e.to) + "_" + (e.label || ""));
            e.id = id;
            newEdges.push(id);
        });

        var currentEdges = edges.getIds();
        var edgesToRemove = currentEdges.filter(function(id) { return !newEdges.includes(String(id)); });
        if (edgesToRemove.length > 0) {
            edges.remove(edgesToRemove);
        }

        dataUpdate.edges.forEach(function (e) {
            if (!edges.get(e.id)) {
                edges.add({
                    id: e.id,
                    from: String(e.from),
                    to: String(e.to),
                    label: e.label,
                    arrows: "to"
                });
            } else {
                edges.update({
                    id: e.id,
                    label: e.label
                });
             }
        });
    }

    // Update stats
    document.getElementById('graph-stats').textContent =
        'Nodes: ' + nodes.length + '  |  Edges: ' + edges.length;
};

window.clearGraph = function() {
    nodes.clear();
    edges.clear();
    document.getElementById('graph-stats').textContent = 'Nodes: 0  |  Edges: 0';
};

initNetwork();
