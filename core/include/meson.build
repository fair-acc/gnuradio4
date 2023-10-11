
graph_dep = declare_dependency(include_directories: '.', dependencies : [fmt_dep, reflcpp_dep, pmt_dep])

header_files = [
    'BufferSkeleton.hpp',
    'Buffer.hpp',
    'CircularBuffer.hpp',
    'ClaimStrategy.hpp',
    'Graph.hpp',
    'BlockTraits.hpp',
    'Block.hpp',
    'PortTraits.hpp',
    'Port.hpp',
    'Sequence.hpp',
    'settings.hpp',
    'typelist.hpp',
    'tag.hpp',
    'transactions.hpp',
    'utils.hpp',
    'WaitStrategy.hpp'
]
install_headers(header_files, subdir : 'graph-prototype', preserve_path: true)