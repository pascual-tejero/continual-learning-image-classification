import os
import xlsxwriter


def save_global_results(dicc_results_test, args):
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
    worksheet = workbook.add_worksheet("Global metrics")  # Create a worksheet

    merge_format = workbook.add_format({
    'bold': 1,
    'border': 1,
    'align': 'center',
    'valign': 'vcenter',
    'fg_color': '#D7E4BC'})

    worksheet.merge_range('A1:H1', f"Global metrics -> Dataset: {args.dataset}", merge_format)

    # Create the headers
    worksheet.write(1, 0, "Test task")
    
    cont_tasks = 2
    cont_avg_task = 2 + args.num_tasks
    for i in range(args.num_tasks):
        for j in range(args.num_tasks):
            worksheet.write(cont_tasks, 0, f"Test accuracy on task {j+1}")
            cont_tasks += 1
        worksheet.write(cont_avg_task, 0, f"Test average accuracy on task {i+1}")
        cont_avg_task += args.num_tasks + 1
        cont_tasks += 1

    # Write the results in the excel file
    col = 0
    for key, value in dicc_results_test.items():
        col += 1
        row = 1

        worksheet.write(row, col, key)
        row += 1

        if key == "Joint datasets":
            for i in range(joint_row):
                worksheet.write(row, col, value[0][1])
                row += 1
            continue
            
        for i in range(args.num_tasks):
            for j in range(args.num_tasks):
                worksheet.write(row, col, value[i][0][j])
                row += 1
            worksheet.write(row, col, value[i][1])
            row += 1
        
        joint_row = row - 2

    workbook.close()  # Close the workbook


# {'Fine-tuning': [[[95.0, 6.3], 50.65], [[22.42, 75.58], 49.0]], 
#  'Joint datasets': [[[84.43], 84.43]], 
#  'Rehearsal 0.1': [[[93.57, 8.72], 51.144999999999996], [[84.5, 73.45], 78.975]], 
#  'Rehearsal 0.3': [[[94.98, 6.7], 50.84], [[94.11, 74.87], 84.49000000000001]], 
#  'Rehearsal 0.5': [[[94.9, 3.95], 49.425000000000004], [[94.57, 74.99], 84.78]], 
#  'EWC': [[[94.62, 4.82], 49.72], [[49.14, 59.88], 54.510000000000005]], 
#  'LwF': [[[95.16, 4.56], 49.86], [[70.03, 54.47], 62.25]]}