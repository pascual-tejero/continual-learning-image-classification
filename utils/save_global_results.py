import os
import xlsxwriter


def save_global_results(dicc_avg_acc, args):
    """
    Create an excel file to save the results of the experiments
    
    """
    # Path to save the results
    path_file = f'./results/{args.exp_name}/global_results_{args.dataset}.xlsx'

    if os.path.exists(path_file):  # If the file exists
        os.remove(path_file)  # Remove the file if it exists

    # Create the excel file
    if os.path.exists(path_file):  # If the file exists
        os.remove(path_file)  # Remove the file if it exists

    workbook = xlsxwriter.Workbook(path_file)  # Create the excel file
    worksheet = workbook.add_worksheet(f"Global metrics ")  # Create a worksheet

    merge_format = workbook.add_format({
    'bold': 1,
    'border': 1,
    'align': 'center',
    'valign': 'vcenter',
    'fg_color': '#D7E4BC'})

    worksheet.merge_range('A1:H1', f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Learning rate: {args.lr}", merge_format)

    # Create the headers
    worksheet.write(1, 0, "Test task")
    
    for i in range(args.num_tasks):
        worksheet.write(i+2, 0, f"Test average accuracy after task {i+1}")

    # Write the results in the excel file
    col = 0
    for key, value in dicc_avg_acc.items():
        col += 1
        row = 1
        worksheet.write(row, col, key)
        row += 1
        for item in value:
            worksheet.write(row, col, item)
            row += 1

    workbook.close()  # Close the workbook